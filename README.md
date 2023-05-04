Download Link: https://assignmentchef.com/product/solved-cs585-hw5-machine-learning
<br>
This last hw is on supervised <strong>machine learning!</strong> As you now know, it’s <strong>data-related</strong> (lots, and lots, and lots of it), after all &#x1f642;

Here is a summary of what you’ll do: <strong>on Google’s <a href="https://colab.research.google.com/">Colab (https://colab.research.google.com/)</a>, train a neural network on differentiating between a cat pic and dog pic, then use the trained network to classify a new (cat-like or dog-like) pic into a cat or dog.</strong> This is a ‘soup-to-nuts’ (start to finish) assignment that will get your feet wet (or plunge you in!), doing ML – a VERY valuable skill – <a href="http://apollo.auto/">training a self-driving car (http://apollo.auto/)</a>, for example, would involve much more complexity, but would be based on the same workflow.

You are going to carry out ‘supervised learning’, as shown in this annotated graphic [from a book on TensorFlow]:

Below are the steps. Have fun!

<ol>

 <li>Use your GMail/GDrive account to log in, go to <a href="https://drive.google.com/">https://drive.google.com/ (https://drive.google.com/),</a> click on the ‘+ New’ button at the top left of the page, look for the ‘Colab’ app [after + New, click on More &gt;, then + Connect more apps] and connect it – this will make the app [which connects to the mighty Google Cloud on the other end!] be able to access (read, write) files and folders in your GDrive.</li>

 <li>You’ll notice that the above step created a folder called Colab Notebooks, inside your GDrive – this is good,because we can keep Colab-related things nicely organized inside that folder. Colab is a cloud environment (maintained by Google), for executing Jupyter ‘notebooks’. A Jupyter notebook (.ipynb extension, ‘Iron Python Notebook’) is a JSON file that contains a mix of two types of “cells” – text cells that have Markdown-formatted text and images, and code cells that contain, well, code &#x1f642; The code can be in <strong>Ju</strong>lia, <strong>Pyt</strong>hon, or <strong>R</strong> (or several other languages, including JavaScript, with appropriate language ‘plugins’ (kernels) installed); for this HW, we’ll use Python notebooks. <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/nb/RealtimeR0.ipynb">Here (nb/RealtimeR0.ipynb)</a> is a COVID-19 notebook. Download it (make sure that ends up on your machine as a .ipynb extension; if your downloading turned it into a .txt, rename it to be .ipynb), then drag and drop it into your Colab GDrive folder. Colab will open the notebook – look through it, to see a mix of text, equations</li>

</ol>

(can contain figures, videos… too), and, most significantly, code. The code in our notebook uses data from <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/pics/RealtimeR0.jpg">https://covidtracking.com/ (https://covidtracking.com/), to calculate ‘R0’ values. In Colab, do ‘Runtime -&gt; Run all’ (pics/RealtimeR0.jpg) to run the code in all the cells; after the code is done running, scroll through, to see the </a>results (several plots). <strong>ALL the computations happened on the cloud, with data fetched by the code, from a</strong>

<strong><a href="https://covidtracking.com/api/v1/states/daily.csv">URL (https://covidtracking.com/api/v1/states/daily.csv</a></strong>

<strong><a href="https://covidtracking.com/api/v1/states/daily.csv">(https://covidtracking.com/api/v1/states/daily.csv)) – pr</a>etty neat!</strong>

<ol start="3">

 <li>Within the Colab Notebooks subdir/folder, create a folder called cats-vs-dogs, for the hw:</li>

</ol>

Now we need DATA [images of cats and dogs] for training and validation, and scripts for training+validation and classifying.

<ol start="4">

 <li>Download <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/data/data.zip">this (data/data.zip)</a> .zip data file (~85MB), unzip it. You’ll see this structure:</li>

</ol>

The train/ folder contains 1000 kitteh images under cats/, and 1000 doggo/pupper ones in dogs/. Have fun, looking at the adorable furballs &#x1f642; Obviously <strong>you</strong> know which is which &#x1f642; A neural network is going to start from scratch, and learn the difference, just based on these 2000 ‘training dataset’ images. The validation/ folder contains 400 images each, of more cats and dogs – these are to feed the trained network, compare its classification answers to the actual answers so we can compute the accuracy of the training (in our code, we do this after each training epoch, to watch the accuracy build up, mostly monotonically). And, live/ is where you’d be placing new (to the NN) images of cats and dogs [that are not in the training or validation datasets], and use their filenames to <strong>ask the network to classify them</strong>: an output of 0 means ‘cat’, 1 means ‘dog’. <strong>Fun!</strong>

Simply drag and drop the data/ folder on to your My Drive/Colab Notebooks/cats-vs-dogs/ area, and wait for about a half hour for the 2800 (2*(1000+400)) images to be uploaded. After that, you should be seeing this [click inside the train/ and validation/ folders to see that the cats and dogs pics have been indeed uploaded]:

<ol start="5">

 <li>OK, <strong>time to train a network</strong>! Download <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/nb/train.ipynb">this (nb/train.ipynb) </a>Jupyter notebook. Drag and drop the notebook into cats-vs-dogs/:</li>

</ol>

Double click on the notebook, that will open it so you can execute the code in the cell(s).

As you can see, it is a VERY short piece of code [not mine, except annotations and mods I made] where a network is set up [starting with ‘model = Sequential()’], and the training is done using it [model.fit_generator()]. In the last line, the RESULTS [learned weights, biases, for each neuron in each layer] are stored on disk as a weights.h5 file [a

<a href="https://en.wikipedia.org/wiki/Hierarchical_Data_Format">.h5 file is binary, in the publicly documented .hd5 file format</a>

<a href="https://en.wikipedia.org/wiki/Hierarchical_Data_Format">(https://en.wikipedia.org/wiki/Hierarchical_Data_Format) (hi</a>erarchical, JSON-like, perfect for storing network weights)].

The code uses the <a href="https://keras.io/">Keras NN library (https://keras.io/),</a> which runs on graph (dataflow) execution backends such TensorFlow(TF), Theano, CNTK [here we are running it over TF via the Google cloud]. With Keras, it is possible to <a href="https://www.datacamp.com/community/blog/keras-cheat-sheet">express NN architectures succintly (https://www.datacamp.com/community/blog/keras-cheat-sheet) </a>– the TF equivalent (or Theano’s etc.) would be more verbose. As a future exercise, you can try coding the model in this hw, directly in TF or Theano or CNTK – you should get the same results.

Before you run the code to kick off the training, note that you will be using GPU acceleration on the cloud (<strong>results in ~10x speedup</strong>) – cool! You’d do this via ‘Edit -&gt; Notebook settings’. In this notebook, this is already set up (by me), but you can verify that it’s set:

When you click on the circular ‘play’ button at the left of the cell, the training will start – here is a sped-up version of what you will get (your numerical values will be different):

0:00 / 0:11

<a href="https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean">The backprop loop runs 50 times (‘epochs’ (https://keras.io/getting-started/faq/#what-does-sample-batch-epochmean)) through all the training data. The acc: column shows the accuracy [how close the training is, to the </a>expected validation/ results], which would be a little over 80% – NOT BAD, for having learned from just 1000 input images for each class!

Click the play button to execute the code! The first time you run it (and anytime after logging out and logging back in), you’d need to authorize Colab to access GDrive – so a message will show up, under the code cell, asking you to click on a link whereby you can log in and provide authorization, and copy and paste the authorization code that appears. Once you do this, the rest of the code (where the training occurs) will start to run.

Scroll down to below the code cell, to watch the training happen. As you can see, it is going to take a short while. After the 50th epoch, we’re all done training (and validating too, which we did 50 times, once at the end of each epoch). <strong>What’s the tangible result, at the end of our training+validating process? It’s a ‘weights.h5’ file!</strong> If you look in your cats-vs-dogs/ folder, it should be there:

<ol start="6">

 <li>Soooo, what exactly [format and content-wise] is in the weights file? You can find out, by downloading HDFView<a href="https://support.hdfgroup.org/products/java/release/download.html">2.14.0, from https://support.hdfgroup.org/products/java/release/download.html</a></li>

</ol>

<a href="https://support.hdfgroup.org/products/java/release/download.html">(https://support.hdfgroup.org/products/java/release/download.html) [grab the b</a>inary, from the ‘HDFView+Object 2.14’ column on the left]. Install, and bring up the program. Download the .h5 file from GDrive to your local area (eg. desktop), then drag and drop it into HDView:

Right-click on weights.h5 at the top-left, and do ‘Expand All’:




Neat! We can see the NN columns, and the biases and weights (kernels) for each. Double click on the bias and kernel items in the second (of the two) dense layers [dense_12, in my case – yours might be named something else], and stagger them so you can see both:

<strong>Computing those floating point numbers is WHAT -EVERY FORM- OF NEURAL NETWORK TRAINING IS</strong>

<strong>ALL ABOUT!</strong> A self-driving car, for example, is also trained the same way, resulting in weights that can classify live traffic data (scary, in my opinion). Here, collectively (taking all layers into account), <strong>it’s those floating point numbers that REPRES</strong><sub>Loading [MathJax]/extensions/MathMenu.js</sub><strong>ENT the network’s “learning” of telling apart cats and dogs!</strong> The “learned” numbers (the

.h5 weights file, actually) can be sent to anyone, who can instantiate a new network (with the same architecture as the one in the training step), and simply re/use the weights in weights.h5 to start classifying cats and dogs right away – no training necessary. The weight arrays represent “catness” and “dogness”, in a sense &#x1f642; We would call the <a href="https://www.wired.com/story/self-driving-cars-power-consumption-nvidia-chip/">network+weights, a ‘pre-trained model’. In a self-driving car, the weights would be copied to the processing hardware (https://www.wired.com/story/self-driving-cars-power-consumption-nvidia-chip/) that resides in t</a>he car. <strong>Q1</strong> [0.5+0.5=1 point]. Submit your weights.h5 file. Also, create a submittable screengrab similar to the above [showing values for the second dense layer (eg. dense_12)]. For fun, click around, examine the arrays in the other layers as well. Again, it’s all these values that are the end result of training, on account of iterating and minimizing classification errors through those epochs.

<ol start="7">

 <li>Now for the fun part – finding out how well our network has learned! Download <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/nb/classify.ipynb">this (nb/classify.ipynb)</a> Jupyter notebook, and upload it to your cats-vs-dogs/ Colab area:</li>

</ol>

When you open classify.ipynb, you can see that it contains Keras code to read the weights file and associate the weights with a new model (which needs to be 100% identical to the one we had set up, to train), then take a new image’s filename as input, and <strong>predict (model.predict()) whether the image is that of a cat [output: 0], or a <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/pics/purple.png">dog [output: 1]!</a></strong><a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/pics/purple.png"> Why 0 for cat and 1 for dog? Because ‘c’ comes before ‘d’ alphabetically [or because (pics/purple.png)</a><a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/pics/purple.png">] &#x1f642;</a>

Supply (upload, into live/) a what1.jpg cat image, and what2.jpg dog image, then execute the cell. Hopefully you’d get a 0, and 1 (for what1.jpg and what2.jpg, respectively). The images can be any resolution (size) and aspect ratio (squarishness), but nearly-square pics would work best. Try this with pics of your pets, your neighbors’, images from a Google search, even your drawings/paintings… <strong>Isn’t this cool? Our little network can classify!</strong>

Just FYI, note that the classification code in classify.ipynb could have simply been inside a new cell in train.ipynb instead. The advantage of multiple code cells inside a notebook, as opposed to multiple code blocks in a script, is that in a notebook, code cells can be independently executed one at a time (usually sequentially) – so if both of our programs were in the same notebook, we would run the training code first (just once), followed by classification (possibly multiple times); a script on the other hand, can’t be re/executed in parts.

<strong>Q2</strong> [2 points]. Create a screenshot that shows the [correct] classification (you’ll also be submitting your what{1,2}.jpg images with this).

What about misclassification? After all, we trained with “just” 1000 (not 1000000) images each, for about an 80% accurate prediction. What if we input ‘difficult’ images, of a cat that looks like it could be labeled a dog, and the other way around? &#x1f642;

<strong><a href="https://www.buzzfeed.com/mjs538/why-corgis-are-the-smartest-animals">Q3</a></strong><a href="https://www.buzzfeed.com/mjs538/why-corgis-are-the-smartest-animals"> [2 points]. Get a ‘Corgi’ image [the world’s smartest (https://www.buzzfeed.com/mjs538/why-corgis-are-thesmartest-animals) dogs!], and a ‘dog-like’ cat image [hint, it’s all about the ears!], upload to live/, attempt to </a>(mis)classify, ie. create incorrect results (where the cat pic outputs a 1, and the dog’s, 0), make a screenshot. Note that you need to edit the code to point myPic and myPic2 to these image filenames. <strong><a href="https://www.google.com/search?q=racoon&amp;rlz=1C1CHBF_enUS723US723&amp;sxsrf=ACYBGNQEveXxDEn19VOiT6UfOQg94Wpe1Q:1574885052359&amp;source=lnms&amp;tbm=isch&amp;sa=X&amp;ved=2ahUKEwjymu6AmIvmAhXbJDQIHUCwAHgQ_AUoAXoECGkQAw&amp;cshid=1574885306579034&amp;biw=1280&amp;bih=578">No-points bonus</a></strong><a href="https://www.google.com/search?q=racoon&amp;rlz=1C1CHBF_enUS723US723&amp;sxsrf=ACYBGNQEveXxDEn19VOiT6UfOQg94Wpe1Q:1574885052359&amp;source=lnms&amp;tbm=isch&amp;sa=X&amp;ved=2ahUKEwjymu6AmIvmAhXbJDQIHUCwAHgQ_AUoAXoECGkQAw&amp;cshid=1574885306579034&amp;biw=1280&amp;bih=578">. Add a third class, eg. racoon (https://www.google.com/search?</a>

<a href="https://www.google.com/search?q=racoon&amp;rlz=1C1CHBF_enUS723US723&amp;sxsrf=ACYBGNQEveXxDEn19VOiT6UfOQg94Wpe1Q:1574885052359&amp;source=lnms&amp;tbm=isch&amp;sa=X&amp;ved=2ahUKEwjymu6AmIvmAhXbJDQIHUCwAHgQ_AUoAXoECGkQAw&amp;cshid=1574885306579034&amp;biw=1280&amp;bih=578">q=racoon&amp;rlz=1C1CHBF_enUS723US723&amp;sxsrf=ACYBGNQEveXxDEn19VOiT6UfOQg94Wpe1Q:1574885052359&amp;source</a>

to the training. You’d ne<sup>Loading [MathJax]/extensions/MathMenu.js </sup>ed to <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/pics/img_scraping.jpg">create (acquire) (pics/img_scraping.jpg) </a>your own training data (500 images might be sufficient, or, even fewer, eg. 300), and <strong>modify the code</strong> just a tiny bit in order to train, and test, on this third class, too. The third class can even be a non-animal, eg. trash (cardboard or platic or metal or glass), ball, furniture, book…

anything at all! <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/nb/train3Classes.ipynb">Here (nb/train3Classes.ipynb)</a> is the training script, and <a href="http://bytes.usc.edu/cs585/s20_db0ds1ml2agi/hw/HW5/nb/classify3Classes.ipynb">here (nb/classify3Classes.ipynb)</a> is the classifying script.

Here’s a checklist of what to submit [<strong>as a single .zip file</strong>]:

<ul>

 <li>h5, and a screenshot from HDFView</li>

 <li>your ‘good’ cat and dog pics, and screenshot that shows proper classification</li>

 <li>your ‘trick’ cat and dog pics, and screenshot that shows misclassification</li>

 <li>if you do the bonus question: your extra data (images) folder for the third class, a ‘new’ third class image you used for classification, a screenshot thatshows correct classification of it, and a README with the code mods you made (this can be actual code with comments, or just a description) – note that the bonus question carries 0 points, so you can skip it if you like</li>

</ul>

All done – hope you had fun, and learned a lot!

<a href="https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks">Note – you can continue using Colab to run all sorts of notebooks (https://github.com/jupyter/jupyter/wiki/Agallery-of-interesting-Jupyter-Notebooks) [on Google’s cloud GPUs!], including ones with TensorFlow, Keras, </a>PyTorch… etc. ML code.

Loading [MathJax]/extensions/MathMenu.js