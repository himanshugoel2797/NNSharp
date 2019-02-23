using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace AnimeImageClassifier
{
    class BooruSearch
    {
        public const string Gel = "https://gelbooru.com";

        struct SearchResult
        {
            public string tags;
            public string file_url;
        }

        private string url;

        private string autocomp_url;
        private string post_url;

        public BooruSearch(string url)
        {
            this.url = url;

            switch (url)
            {
                case Gel:
                    post_url = "index.php?page=dapi&s=post&q=index&json=1&tags=";
                    autocomp_url = "index.php?page=autocomplete&term=";
                    break;
            }
        }

        public string[] AutocompleteTag(string tag)
        {
            string json = "";

            using (var wb = new WebClient())
            {
                var uri = $"{url}/{autocomp_url}{ tag }";
                json = wb.DownloadString(uri);
            }

            string[] results = null;

            try
            {
                results = JsonConvert.DeserializeObject<string[]>(json);

            }
            catch (Exception)
            {
                results = new string[] { "No results found." };
            }

            return results;
        }

        public struct SearchInfo
        {
            public string URL;
            public string[] Tags;
        }

        public SearchInfo[] SearchMultiTag(string[] tag, int idx)
        {
            string json = "";
            //string realTag = BooruSearch.StaticTagAlias(tag);

            using (var wb = new WebClient())
            {
                var uri = $"{url}/{post_url}{ tag[0].ToLower().Replace(' ', '_') }";
                for (int i = 1; i < tag.Length; i++)
                {
                    uri += $"+{tag[i].ToLower().Replace(' ', '_')}";
                }

                json = wb.DownloadString(uri + $"&pid={idx / 100}");
            }


            try
            {
                var t = JsonConvert.DeserializeObject<SearchResult[]>(json);
                var results = new SearchInfo[t.Length];

                for (int i = 0; i < t.Length; i++)
                {
                    results[i].URL = t[i].file_url;
                    results[i].Tags = t[i].tags.Split(' ');
                }
                return results;
            }
            catch (Exception)
            {
                return null;
            }
        }
    }
}
