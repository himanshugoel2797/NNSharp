using NNSharp.ANN;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace NNSharp.Tools
{
    partial class LearningProgressForm : Form
    {
        class NetworkTrainerData
        {
            public int t;
        }

        const float DefaultLearningRate = 0.1f;

        DateTime startTime;
        Dictionary<INetworkTrainer, NetworkTrainerData> trainerData;

        public LearningProgressForm()
        {
            InitializeComponent();
            trainerData = new Dictionary<INetworkTrainer, NetworkTrainerData>();
        }

        private void save_btn_Click(object sender, EventArgs e)
        {
            if (save_net_dialog.ShowDialog() == DialogResult.OK)
            {
                (network_trainer_list.SelectedItem as INetworkTrainer).Save(save_net_dialog.FileName);
            }
        }

        internal void LoadNetwork(INetworkTrainer trainer)
        {
            //Set a default learning rate
            learning_rate_box.Text = DefaultLearningRate.ToString();

            if (!trainerData.ContainsKey(trainer))
            {
                network_trainer_list.Items.Add(trainer);
                network_trainer_list.SelectedItem = trainer;

                var tData = new NetworkTrainerData()
                {
                    t = 0,
                };

                trainerData.Add(trainer, tData);
                loss_chart.Series[0].YValuesPerPoint = trainer.OutputSeriesCount();
            }
        }

        private void test_input_btn_Click(object sender, EventArgs e)
        {
            if (test_input_img_dialog.ShowDialog() == DialogResult.OK)
            {
                //Pass the selected file to the trainer
                (network_trainer_list.SelectedItem as INetworkTrainer).Test(test_input_img_dialog.FileName);

                //TODO: Obtain output to display
            }
        }

        private void network_list_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (network_trainer_list.SelectedItem != null)
            {
                network_list.Items.Clear();

                //Get the networks + layers from the networktrainer

                save_btn.Enabled = true;
                test_input_btn.Enabled = true;
                startstop_btn.Enabled = true;
                learning_rate_box.Enabled = true;
            }
            else
            {
                save_btn.Enabled = false;
                test_input_btn.Enabled = false;
                startstop_btn.Enabled = false;
                learning_rate_box.Enabled = false;
            }
        }

        private void startstop_btn_Click(object sender, EventArgs e)
        {
            startTime = DateTime.Now;
            training_timer.Enabled = !training_timer.Enabled;
            test_input_btn.Enabled = !test_input_btn.Enabled;
        }

        private void training_timer_Tick(object sender, EventArgs e)
        {
            if (network_trainer_list.SelectedItem != null)
            {
                var diff = DateTime.Now.Subtract(startTime);
                time_lbl.Text = $"{diff.Hours,2:00}:{diff.Minutes,2:00}:{diff.Seconds,2:00}:{diff.Milliseconds,3:000}";

                var t = trainerData[(network_trainer_list.SelectedItem as INetworkTrainer)].t++;
                iter_box.Text = t.ToString();
                if ((network_trainer_list.SelectedItem as INetworkTrainer).RunIteration(t, out double[] loss))
                {
                    //Update the associated entry in the chart
                        loss_chart.Series[0].Points.Add(new DataPoint(t, loss));
                }
            }
            else
            {
                training_timer.Enabled = false;
                startstop_btn.Enabled = false;
            }
        }

        private void learning_rate_box_TextChanged(object sender, EventArgs e)
        {
            if (network_trainer_list.SelectedItem != null)
            { 
                if (float.TryParse(learning_rate_box.Text, out float val))
                    (network_trainer_list.SelectedItem as INetworkTrainer).LearningRate = val;
            }
        }
    }
}
