# RoboSpatial-Home

## Data

The data is from [RoboSpatial-Home](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home).

## Process

We refine the prompt of the dataset to make it more clear by replace the post prompt: 

```text
Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points.
```

to the new post prompt:

```text
Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).

Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]

Do not include additional text after this line.
```

## Evaluate

In the evaluation, we extend the official evaluation code to support the new post prompt by extracting the last line of the response as the answer.
