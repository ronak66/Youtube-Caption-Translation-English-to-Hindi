# English to Hindi Machine Translation

Given a YouTube video id, collects the English captions and using a machine translation model, converts these english captions to Hindi and feeds it back to the video.  
It is advised to train the model on a better and a bigger dataset in-order to get acceptable accuracy.

### Steps

1. Run the translate\_youtube\_captions.py code to get the caption and translate it to hindi. This will train the model on the provided dataset and generate predction using the obtained captions.  
Note: Update the youtube video id in the code  
```
$python3 translate_youtube_captions.py
```

This will generate ```eng_to_hindi_translated_script.srt```

2. Run update\_caption.py to update the captions back to the video.  
Note: Update the "client\_secrets\_file", "videoId", "id: "YOUR\_CAPTION\_TRACK\_ID" variable for a succefull update. Also provide the path to the ```eng_to_hindi_translated_script.srt``` in "YOUR_FILE" variable.
```
$python3 update_caption.py
```

### Different language translation

This model is not just for English-to-Hindi translation but can be used for any language translation. You can update the the code by providing the dataset for different language and model will train accordingly.  
Note: New dataset should have same structure as that of current dataset for the code to run smoothly.
