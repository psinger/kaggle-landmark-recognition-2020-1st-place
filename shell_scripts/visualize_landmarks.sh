rm -fr /root/gld_pd/models/visualizations.zip /root/gld_pd/models/visualizations/

python visualize_results.py --config config8
python visualize_results.py --config config9

#zip -r /root/gld_pd/models/visualizations.zip /root/gld_pd/models/visualizations/

