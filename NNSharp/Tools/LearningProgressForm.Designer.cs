namespace NNSharp.Tools
{
    partial class LearningProgressForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.loss_chart = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.learningRateLbl = new System.Windows.Forms.Label();
            this.learning_rate_box = new System.Windows.Forms.TextBox();
            this.network_trainer_list = new System.Windows.Forms.ListBox();
            this.network_list = new System.Windows.Forms.ListBox();
            this.save_btn = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.iter_box = new System.Windows.Forms.TextBox();
            this.netInfoBox = new System.Windows.Forms.GroupBox();
            this.time_lbl = new System.Windows.Forms.Label();
            this.dataset_idx_box = new System.Windows.Forms.TextBox();
            this.dataset_idx_lbl = new System.Windows.Forms.Label();
            this.dataset_sz_box = new System.Windows.Forms.TextBox();
            this.dataset_sz_lbl = new System.Windows.Forms.Label();
            this.test_input_btn = new System.Windows.Forms.Button();
            this.startstop_btn = new System.Windows.Forms.Button();
            this.input_view = new System.Windows.Forms.ListView();
            this.save_net_dialog = new System.Windows.Forms.SaveFileDialog();
            this.training_timer = new System.Windows.Forms.Timer(this.components);
            this.test_input_img_dialog = new System.Windows.Forms.OpenFileDialog();
            ((System.ComponentModel.ISupportInitialize)(this.loss_chart)).BeginInit();
            this.netInfoBox.SuspendLayout();
            this.SuspendLayout();
            // 
            // loss_chart
            // 
            chartArea1.Name = "ChartArea1";
            this.loss_chart.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.loss_chart.Legends.Add(legend1);
            this.loss_chart.Location = new System.Drawing.Point(13, 13);
            this.loss_chart.Name = "loss_chart";
            series1.ChartArea = "ChartArea1";
            series1.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.FastLine;
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.loss_chart.Series.Add(series1);
            this.loss_chart.Size = new System.Drawing.Size(704, 629);
            this.loss_chart.TabIndex = 0;
            this.loss_chart.Text = "Loss Function";
            // 
            // learningRateLbl
            // 
            this.learningRateLbl.AutoSize = true;
            this.learningRateLbl.Location = new System.Drawing.Point(6, 16);
            this.learningRateLbl.Name = "learningRateLbl";
            this.learningRateLbl.Size = new System.Drawing.Size(77, 13);
            this.learningRateLbl.TabIndex = 1;
            this.learningRateLbl.Text = "Learning Rate:";
            // 
            // learning_rate_box
            // 
            this.learning_rate_box.Enabled = false;
            this.learning_rate_box.Location = new System.Drawing.Point(89, 13);
            this.learning_rate_box.Name = "learning_rate_box";
            this.learning_rate_box.Size = new System.Drawing.Size(89, 20);
            this.learning_rate_box.TabIndex = 2;
            this.learning_rate_box.TextChanged += new System.EventHandler(this.learning_rate_box_TextChanged);
            // 
            // network_trainer_list
            // 
            this.network_trainer_list.FormattingEnabled = true;
            this.network_trainer_list.Location = new System.Drawing.Point(723, 13);
            this.network_trainer_list.Name = "network_trainer_list";
            this.network_trainer_list.Size = new System.Drawing.Size(186, 160);
            this.network_trainer_list.TabIndex = 3;
            this.network_trainer_list.SelectedIndexChanged += new System.EventHandler(this.network_list_SelectedIndexChanged);
            // 
            // network_list
            // 
            this.network_list.FormattingEnabled = true;
            this.network_list.Location = new System.Drawing.Point(9, 211);
            this.network_list.Name = "network_list";
            this.network_list.Size = new System.Drawing.Size(169, 238);
            this.network_list.TabIndex = 4;
            // 
            // save_btn
            // 
            this.save_btn.Enabled = false;
            this.save_btn.Location = new System.Drawing.Point(9, 153);
            this.save_btn.Name = "save_btn";
            this.save_btn.Size = new System.Drawing.Size(74, 23);
            this.save_btn.TabIndex = 5;
            this.save_btn.Text = "Save";
            this.save_btn.UseVisualStyleBackColor = true;
            this.save_btn.Click += new System.EventHandler(this.save_btn_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 41);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(53, 13);
            this.label1.TabIndex = 6;
            this.label1.Text = "Iterations:";
            // 
            // iter_box
            // 
            this.iter_box.Enabled = false;
            this.iter_box.Location = new System.Drawing.Point(89, 38);
            this.iter_box.Name = "iter_box";
            this.iter_box.Size = new System.Drawing.Size(89, 20);
            this.iter_box.TabIndex = 7;
            // 
            // netInfoBox
            // 
            this.netInfoBox.Controls.Add(this.time_lbl);
            this.netInfoBox.Controls.Add(this.dataset_idx_box);
            this.netInfoBox.Controls.Add(this.dataset_idx_lbl);
            this.netInfoBox.Controls.Add(this.dataset_sz_box);
            this.netInfoBox.Controls.Add(this.dataset_sz_lbl);
            this.netInfoBox.Controls.Add(this.test_input_btn);
            this.netInfoBox.Controls.Add(this.startstop_btn);
            this.netInfoBox.Controls.Add(this.learningRateLbl);
            this.netInfoBox.Controls.Add(this.network_list);
            this.netInfoBox.Controls.Add(this.iter_box);
            this.netInfoBox.Controls.Add(this.learning_rate_box);
            this.netInfoBox.Controls.Add(this.label1);
            this.netInfoBox.Controls.Add(this.save_btn);
            this.netInfoBox.Location = new System.Drawing.Point(723, 187);
            this.netInfoBox.Name = "netInfoBox";
            this.netInfoBox.Size = new System.Drawing.Size(186, 455);
            this.netInfoBox.TabIndex = 8;
            this.netInfoBox.TabStop = false;
            this.netInfoBox.Text = "Network Info";
            // 
            // time_lbl
            // 
            this.time_lbl.AutoSize = true;
            this.time_lbl.Location = new System.Drawing.Point(89, 158);
            this.time_lbl.Name = "time_lbl";
            this.time_lbl.Size = new System.Drawing.Size(53, 13);
            this.time_lbl.TabIndex = 17;
            this.time_lbl.Text = "Run Time";
            // 
            // dataset_idx_box
            // 
            this.dataset_idx_box.Enabled = false;
            this.dataset_idx_box.Location = new System.Drawing.Point(89, 89);
            this.dataset_idx_box.Name = "dataset_idx_box";
            this.dataset_idx_box.Size = new System.Drawing.Size(89, 20);
            this.dataset_idx_box.TabIndex = 15;
            // 
            // dataset_idx_lbl
            // 
            this.dataset_idx_lbl.AutoSize = true;
            this.dataset_idx_lbl.Location = new System.Drawing.Point(7, 92);
            this.dataset_idx_lbl.Name = "dataset_idx_lbl";
            this.dataset_idx_lbl.Size = new System.Drawing.Size(76, 13);
            this.dataset_idx_lbl.TabIndex = 14;
            this.dataset_idx_lbl.Text = "Dataset Index:";
            // 
            // dataset_sz_box
            // 
            this.dataset_sz_box.Enabled = false;
            this.dataset_sz_box.Location = new System.Drawing.Point(89, 64);
            this.dataset_sz_box.Name = "dataset_sz_box";
            this.dataset_sz_box.Size = new System.Drawing.Size(89, 20);
            this.dataset_sz_box.TabIndex = 13;
            // 
            // dataset_sz_lbl
            // 
            this.dataset_sz_lbl.AutoSize = true;
            this.dataset_sz_lbl.Location = new System.Drawing.Point(6, 67);
            this.dataset_sz_lbl.Name = "dataset_sz_lbl";
            this.dataset_sz_lbl.Size = new System.Drawing.Size(70, 13);
            this.dataset_sz_lbl.TabIndex = 12;
            this.dataset_sz_lbl.Text = "Dataset Size:";
            // 
            // test_input_btn
            // 
            this.test_input_btn.Enabled = false;
            this.test_input_btn.Location = new System.Drawing.Point(89, 182);
            this.test_input_btn.Name = "test_input_btn";
            this.test_input_btn.Size = new System.Drawing.Size(89, 23);
            this.test_input_btn.TabIndex = 9;
            this.test_input_btn.Text = "Test Input";
            this.test_input_btn.UseVisualStyleBackColor = true;
            this.test_input_btn.Click += new System.EventHandler(this.test_input_btn_Click);
            // 
            // startstop_btn
            // 
            this.startstop_btn.Enabled = false;
            this.startstop_btn.Location = new System.Drawing.Point(9, 182);
            this.startstop_btn.Name = "startstop_btn";
            this.startstop_btn.Size = new System.Drawing.Size(74, 23);
            this.startstop_btn.TabIndex = 8;
            this.startstop_btn.Text = "Start/Stop";
            this.startstop_btn.UseVisualStyleBackColor = true;
            this.startstop_btn.Click += new System.EventHandler(this.startstop_btn_Click);
            // 
            // input_view
            // 
            this.input_view.Location = new System.Drawing.Point(915, 12);
            this.input_view.Name = "input_view";
            this.input_view.Size = new System.Drawing.Size(518, 624);
            this.input_view.TabIndex = 10;
            this.input_view.UseCompatibleStateImageBehavior = false;
            // 
            // save_net_dialog
            // 
            this.save_net_dialog.Title = "Save Neural Network";
            // 
            // training_timer
            // 
            this.training_timer.Tick += new System.EventHandler(this.training_timer_Tick);
            // 
            // test_input_img_dialog
            // 
            this.test_input_img_dialog.Title = "Select Test Input Image";
            // 
            // LearningProgressForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1445, 654);
            this.Controls.Add(this.input_view);
            this.Controls.Add(this.netInfoBox);
            this.Controls.Add(this.network_trainer_list);
            this.Controls.Add(this.loss_chart);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "LearningProgressForm";
            this.ShowIcon = false;
            this.Text = "Learning Progress";
            ((System.ComponentModel.ISupportInitialize)(this.loss_chart)).EndInit();
            this.netInfoBox.ResumeLayout(false);
            this.netInfoBox.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.DataVisualization.Charting.Chart loss_chart;
        private System.Windows.Forms.Label learningRateLbl;
        private System.Windows.Forms.TextBox learning_rate_box;
        private System.Windows.Forms.ListBox network_trainer_list;
        private System.Windows.Forms.ListBox network_list;
        private System.Windows.Forms.Button save_btn;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox iter_box;
        private System.Windows.Forms.GroupBox netInfoBox;
        private System.Windows.Forms.ListView input_view;
        private System.Windows.Forms.Button test_input_btn;
        private System.Windows.Forms.Button startstop_btn;
        private System.Windows.Forms.Label dataset_sz_lbl;
        private System.Windows.Forms.TextBox dataset_idx_box;
        private System.Windows.Forms.Label dataset_idx_lbl;
        private System.Windows.Forms.TextBox dataset_sz_box;
        private System.Windows.Forms.SaveFileDialog save_net_dialog;
        private System.Windows.Forms.Timer training_timer;
        private System.Windows.Forms.OpenFileDialog test_input_img_dialog;
        private System.Windows.Forms.Label time_lbl;
    }
}