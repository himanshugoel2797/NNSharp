using NNSharp.ANN;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace AnimeImageClassifier
{
    class BooruDatasetBuilder
    {
        BooruSearch booruSearch;

        List<string> localTags, globalTags;
        List<string> names;

        public BooruDatasetBuilder()
        {
            names = new List<string>();
            booruSearch = new BooruSearch(BooruSearch.Gel);
            localTags = new List<string>();
            globalTags = new List<string>();
        }

        public void AddLocalTag(string tag)
        {
            localTags.Add(tag);
        }

        public void AddGlobalTag(string tag)
        {
            globalTags.Add(tag);
        }

        public void Download(int per_tag_max, string path)
        {
            Directory.CreateDirectory(Path.Combine(path, "Images"));
            Directory.CreateDirectory(Path.Combine(path, "Tags"));

            var glbl_tags = new string[globalTags.Count + 1];
            for (int i = 0; i < globalTags.Count; i++)
                glbl_tags[i] = globalTags[i];

            int idx = 0;

            XmlSerializer serializer = new XmlSerializer(typeof(string[]));
            for (int i = 0; i < localTags.Count; i++)
            {
                names.Clear();
                glbl_tags[globalTags.Count] = localTags[i];

                localTags[i] = localTags[i].Replace(':', '_');

                if (Directory.Exists(Path.Combine(path, "Images", localTags[i].Replace('/', '_'))))
                    continue;

                Console.WriteLine("Current Tag: " + localTags[i]);
                Directory.CreateDirectory(Path.Combine(path, "Images", localTags[i].Replace('/', '_')));
                Directory.CreateDirectory(Path.Combine(path, "Tags", localTags[i].Replace('/', '_')));

                for (int j = 0; j < per_tag_max;)
                {
                    var results = booruSearch.SearchMultiTag(glbl_tags, j);
                    if (results == null | results.Length == 0) break;

                    for (int q = 0; q < results.Length; q++)
                    {
                        Console.WriteLine("Downloading Image #" + j);

                        if (!names.Contains(results[q].URL))
                        {
                            names.Add(results[q].URL);

                            byte[] data = null;
                            using (var wb = new WebClient())
                                data = wb.DownloadData(results[q].URL);

                            using (MemoryStream strm = new MemoryStream(data))
                            using (var bmp = new Bitmap(strm))
                                bmp.Save(Path.Combine(path, $"Images", localTags[i].Replace('/', '_'), $"{j}.png"));

                            using (var writer = File.OpenWrite(Path.Combine(path, $"Tags", localTags[i].Replace('/', '_'), $"{j}.xml")))
                                serializer.Serialize(writer, results[q].Tags);
                        }

                        j++;
                    }
                }
            }
        }
        /*
        public IDataset GetDataset(string srcPath, string smallPath, int res, int max_imgs_per_class)
        {
            LabeledFileImageSet imageSet = new LabeledFileImageSet(smallPath, res, max_imgs_per_class * localTags.Count, localTags.Count, 0);

            for (int i = 0; i < localTags.Count; i++)
            {
                var tags = new float[localTags.Count];
                //for (int q = 0; q < tags.Length; q++)
                //    tags[q] = -1;

                tags[i] = 1;

                var files = Directory.EnumerateFiles(Path.Combine(srcPath, "Images", localTags[i])).ToArray();
                for (int j = 0; j < max_imgs_per_class; j++)
                    imageSet.AddFile(files[j], tags);
            }

            imageSet.Initialize();
            return imageSet;
        }*/
    }
}
