import cv2
import numpy as np

def find_board(img):
  blur = cv2.GaussianBlur(img, (3,3), 0)
  edges = cv2.Canny(blur, 50, 150, apertureSize=3)
  cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1] 
  
  x,y,w,h = cv2.boundingRect(cnts[np.argmax([cv2.contourArea(c) for c in cnts])]) # get largest box coords
  board_side_len = w if w%8 == 0 else h if h%8 == 0 else (w+8//2) // 8*8 # compute best board dim
  return img[y:y+board_side_len, x:x+board_side_len]
  
if __name__ == "__main__":
  img = cv2.imread("screen_3.png")
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # resize
  new_width = int(img.shape[1] * 0.3)
  new_height = int(img.shape[0] * 0.3)
  img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

  board = find_board(img)
  splited_board = np.array([np.hsplit(x, 8) for x in np.vsplit(board, 8)])

  case = cv2.resize(splited_board[0, 3], (32, 32), interpolation=cv2.INTER_LINEAR)

  cv2.imshow("board", case)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
