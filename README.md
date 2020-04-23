# English to Hindi Machine Translation

Given a youtube id, gets the English captions and using a machine trasnlation model, converts these english captions to Hindi and feeds it back to the video.

### Steps

1. Run the translate\_youtube\_captions.py code to get the caption and translate it to hindi.  
Note: Update the youtube video id in the code  
```
$python3 translate_youtube_captions.py
```

2. Run update\_caption.py to update the captions back to the video.  
Note: Update the "client\_secrets\_file", "videoId", "id: "YOUR\_CAPTION\_TRACK\_ID" variable for a succefull update  
```
$python3 update_caption.py
```

### Different language translation

This model is not just for English-to-Hindi translation but can be used for any language translation. You can update the the code by providing the dataset for different language and model will train accordingly.  
Note: New dataset should have same structure as that of current dataset for the code to run smoothly.
