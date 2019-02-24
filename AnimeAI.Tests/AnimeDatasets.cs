using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnimeAI.Tests
{
    public class AnimeDatasets
    {
        //const string TrainingDataPath = @"I:\Datasets\Lewds\Pictures";
        //const string TrainingDataPath_SMALL = @"I:\Datasets\Lewds\Pictures\small";
        readonly string TrainingDataPath = @"I:\Datasets\anime-faces\emilia_(re_zero)";
        readonly string TrainingDataPath_SMALL = @"I:\Datasets\anime-faces\emilia_small2";
        //const string TrainingDataPath = @"I:\Datasets\anime-faces\combined";
        //const string TrainingDataPath_SMALL = @"I:\Datasets\anime-faces\combined_small";
        public List<string> TrainingFiles;

        public int Side;
        public int InputSize;

        public AnimeDatasets(int side, string path, string path_small)
        {
            Side = side;
            InputSize = Side * Side * 3;

            TrainingDataPath = path;
            TrainingDataPath_SMALL = path_small;
        }

        #region Image Manipulation Routines
        static Random rsz_rng = new Random(1);
        public static Bitmap ScaleImage(Image image, int width, int height)
        {
            float aspectRatio = (float)image.Width / image.Height;

            float height_var = (height - (int)(height * aspectRatio)) * 0.8f;
            float width_var = (width - (int)(width / aspectRatio)) * 0.8f;

            var destRect = new Rectangle((int)(height_var * rsz_rng.NextDouble()), 0, (int)(height * aspectRatio), width);
            if (aspectRatio > 1.0f)
                destRect = new Rectangle(0, (int)(width_var * rsz_rng.NextDouble()), height, (int)(width / aspectRatio));
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static string[] CutImage(Image image, int width, int height, string dst, string filename)
        {
            List<string> cut_paths = new List<string>();

            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            //destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            for (int y = 0; y < image.Height - height; y += height / 2)
                for (int x = 0; x < image.Width - width; x += width / 2)
                {
                    using (var graphics = Graphics.FromImage(destImage))
                    {
                        graphics.CompositingMode = CompositingMode.SourceCopy;
                        graphics.CompositingQuality = CompositingQuality.HighQuality;
                        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        graphics.SmoothingMode = SmoothingMode.HighQuality;
                        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                        using (var wrapMode = new ImageAttributes())
                        {
                            wrapMode.SetWrapMode(WrapMode.Clamp);
                            //wrapMode.
                            graphics.DrawImage(image, destRect, x, y, width, height, GraphicsUnit.Pixel, wrapMode);
                        }
                    }

                    string path = Path.Combine(dst, filename + "_" + x + "_" + y + ".png");
                    cut_paths.Add(path);
                    destImage.Save(path);
                }

            destImage.Dispose();
            return cut_paths.ToArray();
        }
        #endregion

        public void InitializeDataset()
        {
            TrainingFiles = new List<string>();

            Directory.CreateDirectory(TrainingDataPath_SMALL);

            var files = Directory.EnumerateFiles(TrainingDataPath).ToArray();
            Parallel.For(0, files.Length, (i) =>
            {
                if (new string[] { ".png", ".jpg" }.Contains(Path.GetExtension(files[i])))
                {
                    var smallPath = Path.ChangeExtension(Path.Combine(TrainingDataPath_SMALL, Path.GetFileName(files[i])), "png");

                    if (!File.Exists(smallPath))
                    {
                        var bmp = new Bitmap(files[i]);
                        var rsz_bmp = ResizeImage(bmp, Side, Side);
                        rsz_bmp.Save(smallPath);
                        rsz_bmp.Dispose();

                        //var cut_src = ResizeImage(bmp, Side * 2, (int)(Side * 2 * bmp.Height / (float)bmp.Width));
                        //TrainingFiles.AddRange(CutImage(cut_src, Side, Side, TrainingDataPath_SMALL, Path.GetFileNameWithoutExtension(files[i])));
                        bmp.Dispose();
                    }
                }
            });

            if (TrainingFiles.Count == 0)
            {
                Random rng = new Random(0);
                TrainingFiles.AddRange(Directory.EnumerateFiles(TrainingDataPath_SMALL));
                while (TrainingFiles.Count > 5000)
                {
                    TrainingFiles.RemoveAt(TrainingFiles.Count - 1);
                }
            }
            else
            {
                var img = new float[InputSize];
                for (int idx = 0; idx < TrainingFiles.Count; idx++)
                {
                    var bmp = new Bitmap(TrainingFiles[idx]);
                    bool isSame = true;

                    var pxl0 = bmp.GetPixel(0, 0);
                    byte pR = pxl0.R;
                    byte pG = pxl0.G;
                    byte pB = pxl0.B;

                    int i = 0;
                    for (int h = 0; h < bmp.Height; h++)
                        for (int w = 0; w < bmp.Width; w++)
                        {
                            var pxl = bmp.GetPixel(w, h);
                            if (pR != pxl.R | pG != pxl.G | pB != pxl.B)
                            {
                                isSame = false;
                                break;
                            }
                            pR = pxl.R;
                            pG = pxl.G;
                            pB = pxl.B;

                            img[i++] = pxl.R / 255.0f;
                            img[i++] = pxl.G / 255.0f;
                            img[i++] = pxl.B / 255.0f;
                        }

                    bmp.Dispose();

                    if (isSame)
                    {
                        File.Delete(TrainingFiles[idx]);
                        TrainingFiles.RemoveAt(idx);
                        idx--;
                    }
                }
            }
        }

        public void LoadImage(string file, float[] img)
        {
            var bmp = new Bitmap(file);

            int i = 0;
            for (int h = 0; h < bmp.Height; h++)
                for (int w = 0; w < bmp.Width; w++)
                {
                    var pxl = bmp.GetPixel(w, h);
                    img[h * bmp.Width + w] = pxl.R / 255.0f;
                    img[bmp.Width * bmp.Height + h * bmp.Width + w] = pxl.G / 255.0f;
                    img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] = pxl.B / 255.0f;
                }


            for (int q = 0; q < img.Length; q++)
            {
                img[q] = img[q] * 2.0f - 1.0f;
            }

            bmp.Dispose();
        }

        public void SaveImage(string file, float[] img)
        {
            var bmp = new Bitmap(Side, Side);


            float max = float.MinValue;
            float min = float.MaxValue;

            for (int i = 0; i < img.Length; i++)
            {
                if (img[i] > max)
                    max = img[i];

                if (img[i] < min)
                    min = img[i];
                /*
                img[i] /= 255.0f;
                if (img[i] > 1.0f)
                    img[i] = 1.0f;

                if (img[i] < 0.0f)
                    img[i] = 0.0f;*/
            }

            try
            {
                for (int h = 0; h < bmp.Height; h++)
                    for (int w = 0; w < bmp.Width; w++)
                    {
                        //img[h * bmp.Width + w] = (img[h * bmp.Width + w] - min) / (max - min);
                        //img[bmp.Width * bmp.Height + h * bmp.Width + w] = (img[bmp.Width * bmp.Height + h * bmp.Width + w] - min) / (max - min);
                        //img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] = (img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] - min) / (max - min);

                        bmp.SetPixel(w, h, Color.FromArgb((int)((img[h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f), (int)((img[bmp.Width * bmp.Height + h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f), (int)((img[bmp.Width * bmp.Height * 2 + h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f)));
                    }
            }
            catch (Exception) { }

            bmp.Save(file);
            bmp.Dispose();
        }

        public void SaveImage(string file, float[] img, int offset, int Side)
        {
            var bmp = new Bitmap(Side, Side);

            float max = float.MinValue;
            float min = float.MaxValue;

            for (int i = offset; i < offset + Side * Side; i++)
            {
                if (img[i] < -1) img[i] = -1;
                if (img[i] > 1) img[i] = 1;

                img[i] = (float)Math.Pow(img[i], 2);

                if (img[i] > max)
                    max = img[i];

                if (img[i] < min)
                    min = img[i];
            }

            try
            {
                for (int h = 0; h < bmp.Height; h++)
                    for (int w = 0; w < bmp.Width; w++)
                    {
                        //img[offset + h * bmp.Width + w] = (img[offset + h * bmp.Width + w] - min) / (max - min);
                        //if(img[offset + h * bmp.Width + w])

                        bmp.SetPixel(w, h, Color.FromArgb((int)((img[offset + h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f), (int)((img[offset + h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f), (int)((img[offset + h * bmp.Width + w] * 0.5f + 0.5f) * 255.0f)));
                    }
            }
            catch (Exception) { }

            bmp.Save(file);
            bmp.Dispose();
        }
    }
}
