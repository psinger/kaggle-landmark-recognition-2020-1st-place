<div align="center" markdown>
<img src="https://i.imgur.com/ndhSKM8.jpg"/>


# AI Recommendations

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Demo-Video">Demo Video</a> •
  <a href="#Results">Results</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/gl-metric-learning/supervisely/similarity-calculator)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/gl-metric-learning)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/gl-metric-learning/supervisely/similarity-calculator)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/gl-metric-learning/supervisely/similarity-calculator)](https://supervise.ly)

</div>

# Overview

Application calculates cosine similarity between reference database and incoming embeddings.  
It returns recommended items from reference database with their probabilities (cosine similarity score).

Application key points:
- Return recommended item from reference database [using API](https://github.com/supervisely-ecosystem/gl-metric-learning/blob/main/supervisely/similarity-calculator/src/demo_api_requests.py)
- Load previously calculated embeddings from [Embeddings Calculator](https://ecosystem.supervise.ly/apps/gl-metric-learning/supervisely/calculator) application to RAM
- Allows dynamically updates reference database in RAM

# How to Run

### 1. Add [AI Recommendations](https://ecosystem.supervise.ly/apps/gl-metric-learning/supervisely/similarity-calculator) to your team from Ecosystem.
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/similarity-calculator" src="https://i.imgur.com/1QpAfy2.png" width="350px" style='padding-bottom: 20px'/>  

### 2. Run app from the context menu of **Images Project**:
ℹ️ You can use [Embeddings Calculator](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/gl-metric-learning/supervisely/calculator) application to get Images Project in suitable format
<img src="https://i.imgur.com/XHZ4OZq.png" width="100%"/>


# How to Use

1. Select checkpoint with which you computed the embeddings for the selected project
2. Load selected embeddings

<img src="https://i.imgur.com/m8jmQIv.png" width="100%"/>

# Demo Video
<a data-key="sly-embeded-video-link" href="https://youtu.be/okFQDSJTgYk" data-video-code="okFQDSJTgYk">
    <img src="https://i.imgur.com/qchPUuB.png" alt="SLY_EMBEDED_VIDEO_LINK"  width="70%">
</a>

# Results

Embeddings have been loaded to RAM and are ready to go.  
The application will continue to run in server mode and wait for incoming requests.

[**requests examples**](https://github.com/supervisely-ecosystem/gl-metric-learning/blob/main/supervisely/similarity-calculator/src/demo_api_requests.py)

<img src="https://i.imgur.com/D9pXsm9.png" width="100%"/>
