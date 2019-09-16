# **Finding Lane Lines on the Road** 


## 1. steps

### 1) convert color image to gray image.
### 2) apply Gaussian blur to deal with noise
### 3) apply canny edge detetion
### 4) create a region of interest, and mask rest of the image.
### 5) apply Hough transform and get lane lines.
### 6) for dashed lane markings, setp 5 will give only dash output. So did this
####      a) first separated left and right side markings, using slope.
####      b) discarded left/right lines whose slope is out of range (Hueristic)
####      c) used numpy polyfit to get best fit left/right line
####      d) extrapolated left/right lines for the whole region of interest 
####      e) still for yellow lines there were few wrong detected lines. So used polygon.
#####        e.1) created a polygon from region of interest
#####        e.2) if line is out of the ROI then discarded it
### 7) finally draw it


## 2. Identify potential shortcomings with your current pipeline

### 1) still it does not work for curved lanes.
### 2) lots of magic numbers (hardcoded values), which makes it very difficult to rely upon in unknown situation.
### 3) since hardcoded values are so many, its a pain to tune them for satisfactory output
### 5) shadows, road craks are also not handled.

##### and list goes on ...


## 3. Suggest possible improvements to your pipeline

### 1) reducing hardcoded values.
### 2) to work for curvy lanes, we should have many lines, connected together
### 3) to remove shadows, we should detect them and dicard.
### 4) same for road crakes.
### 5) road lines can be reflected from other surface (other shiny car etc.). that has to be delt with

##### and list goes on ...