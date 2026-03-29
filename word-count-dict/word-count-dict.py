def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code 
    dict={}
    for i in range(len(sentences)):
        for j in sentences[i]:
            if j in dict:
                dict[j]+=1
            else:
                dict[j]=1
    return dict