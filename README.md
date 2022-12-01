**CS1470/2470 Final Project**

**Members: **

Dustin Wu, Milanca Wang  

CS1470/2470 Final Project Reflection

**Members:**

Dustin Wu, Milanca Wang  

**Introduction: **

One distinct memory from high school that many studying in the United States have in common is preparing for standardized tests, primarily the SAT and the ACT. Many will comment that the most challenging part of these tests are the reading comprehension sections, where students must read, comprehend, and answer multiple-choice questions about long passages. This is for good reason: the passages are oftentimes difficult to understand, being sourced anywhere from old English literature to modern day research. Furthermore, the questions demand a deep understanding of the content, as some questions will ask about the overall ideas and intentions of a written work while others will focus on the minutiae, for instance of the meaning of a specific word in the context of a sentence.

Given that these tests were such a relevant part of our educational careers and given that they pose an interesting yet formidable challenge, we believe that solving these questions is an exciting problem to tackle. This problem is a supervised learning, natural language processing task, and our model will have to understand the content of passages in order to achieve anything close to resembling success.

**Challenges:**

One of the biggest challenges has been working with pretrained NLP models. We decided that relying on pretrained models was necessary, as the difficulty of NLP problems means that there is an inherent need for a large, complex model trained on mountains worth of data. Given the short time period that we had to complete this project, we decided that it would be much more effective to start from a pretrained model. This isn’t to say that our entire project merely involves finetuning a pretrained model; one of the weaknesses of NLP models is that they can only take in a limited number of characters as input, and this limit is far less than the length of a typical SAT/ACT passage. To address this, we trained an auxiliary model that takes questions as input and identifies the portion of text that is best suited for answering the question. Doing so was difficult because well-documented pretrained models for NLP are difficult to come by. Huggingface had by far the most resources for doing this, however the documentation was somewhat lacking and it was especially difficult to work with the pretrained model that we chose, because Huggingface primarily uses Pytorch so we had to rapidly learn how to use the new framework.

**Insights:**

We have been pleasantly surprised with the performance of our initial model. We have finetuned a pretrained Roberta model, which is a modified version of the well-known BERT transformer, on SAT and ACT passages and questions extracted from practice test pdfs, and it has had over a 40% accuracy on our validation set of 91 SAT and ACT questions. This far exceeds our expectations, as given the difficulty of these questions, we had expected the model to barely scratch a 30% accuracy. This result speaks greatly to the power of the pretrained model that we used.

**Plan:**

While we are pleased with how our initial model has been performing, 40% accuracy is not very impressive by human test-taker standards, and we have a lot of room for improvement. Our initial model naively only considers the first 200 characters of each passage, so its performance is expected to improve significantly when the model is given a more holistic representation of the passage.

We are currently working on building and training a model for deriving the optimal “context window” from a passage for answering a given input question. However, due to the difficulties that have arisen with having to use Pytorch instead of Tensorflow, we have some work to do before this auxiliary model is ready for use to work in conjunction with our initial model. In particular, we need to spend more time working out the kinks of building a data pipeline for feeding input questions and passage sections into the model. However, we have already made significant progress in getting this part of the project done and foresee completing everything that we need to before the final deadline. 

Once this auxiliary context window model is trained, we plan to use its output, which is the section of passage that is best suited for answering a question, as the input for our initial multiple choice question-answering model. We predict that the accuracy of this combined model on the validation set will improve, and with a baseline of 40% accuracy we are optimistic that our results on the test set, which we have held in reserve up to this point, will be equally impressive.

One distinct memory from high school that many studying in the United States have in common is preparing for standardized tests, primarily the SAT and the ACT. Many will comment that the most challenging part of these tests are the reading comprehension sections, where students must read, comprehend, and answer multiple-choice questions about long passages. This is for good reason: the passages are oftentimes difficult to understand, being sourced anywhere from old English literature to modern day research. Furthermore, the questions demand a deep understanding of the content, as some questions will ask about the overall ideas and intentions of a written work while others will focus on the minutiae, for instance of the meaning of a specific word in the context of a sentence.

Given that these tests were such a relevant part of our educational careers and given that they pose an interesting yet formidable challenge, we believe that solving these questions is an exciting problem to tackle. This problem is a supervised learning, natural language processing task, and our model will have to understand the content of passages in order to achieve anything close to resembling success.

**Challenges:**

One of the biggest challenges has been working with pretrained NLP models. We decided that relying on pretrained models was necessary, as the difficulty of NLP problems means that there is an inherent need for a large, complex model trained on mountains worth of data. Given the short time period that we had to complete this project, we decided that it would be much more effective to start from a pretrained model. This isn’t to say that our entire project merely involves finetuning a pretrained model; one of the weaknesses of NLP models is that they can only take in a limited number of characters as input, and this limit is far less than the length of a typical SAT/ACT passage. To address this, we trained an auxiliary model that takes questions as input and identifies the portion of text that is best suited for answering the question. Doing so was difficult because well-documented pretrained models for NLP are difficult to come by. Huggingface had by far the most resources for doing this, however the documentation was somewhat lacking and it was especially difficult to work with the pretrained model that we chose, because Huggingface primarily uses Pytorch so we had to rapidly learn how to use the new framework.

**Insights:**

We have been pleasantly surprised with the performance of our initial model. We have finetuned a pretrained Roberta model, which is a modified version of the well-known BERT transformer, on SAT and ACT passages and questions extracted from practice test pdfs, and it has had over a 40% accuracy on our validation set of 91 SAT and ACT questions. This far exceeds our expectations, as given the difficulty of these questions, we had expected the model to barely scratch a 30% accuracy. This result speaks greatly to the power of the pretrained model that we used.

**Plan:**

While we are pleased with how our initial model has been performing, 40% accuracy is not very impressive by human test-taker standards, and we have a lot of room for improvement. Our initial model naively only considers the first 200 characters of each passage, so its performance is expected to improve significantly when the model is given a more holistic representation of the passage.

We are currently working on building and training a model for deriving the optimal “context window” from a passage for answering a given input question. However, due to the difficulties that have arisen with having to use Pytorch instead of Tensorflow, we have some work to do before this auxiliary model is ready for use to work in conjunction with our initial model. In particular, we need to spend more time working out the kinks of building a data pipeline for feeding input questions and passage sections into the model. However, we have already made significant progress in getting this part of the project done and foresee completing everything that we need to before the final deadline. 

Once this auxiliary context window model is trained, we plan to use its output, which is the section of passage that is best suited for answering a question, as the input for our initial multiple choice question-answering model. We predict that the accuracy of this combined model on the validation set will improve, and with a baseline of 40% accuracy we are optimistic that our results on the test set, which we have held in reserve up to this point, will be equally impressive.
