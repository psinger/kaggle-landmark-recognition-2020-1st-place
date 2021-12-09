<div align="center" markdown>
<img src=""/>


# Metric Learning Labeling Tool

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Results">Results</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Application calculates cosine similarity between reference database and incoming embeddings.  
It returns recommended items from reference database with their probabilities (cosine similarity score).

Application key points:
- Preview top@N predicted items from reference database
- Label new data by predicted items
- Every new-labled item can be added to reference database
- Review mode for review labeled objects
- Copy mode for fast labeling
- Browse reference database

# How to Run
1. Add [Metric Learning Labeling Tool](https://ecosystem.supervise.ly/apps/gl-metric-learning/supervisely/labeling-tool) to your team from Ecosystem.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool" src="https://i.imgur.com/P3oE9Re.png" width="350px" style='padding-bottom: 20px'/>  

2. Run app from the apps tab in **Image Annotation Tool**:

<img src="https://i.imgur.com/JCyFsqP.png" width="100%"/>

**Note**: Move apps tab to the left for a better experience

<img src="https://i.imgur.com/OSbrDm1.png" width="100%"/>

# How to Use

1. Select checkpoint with which you computed the embeddings for the selected project
2. Load selected embeddings

<img src="" width="100%"/>

# Results

Embeddings have been loaded to RAM and are ready to go.  
The application will continue to run in server mode and wait for incoming requests.

<img src="" width="100%"/>
