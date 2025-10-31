import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def createAnim(obsVec):
    fig, ax = plt.subplots()
    ax.axis('off') # Hide the axes

    # --- 3. Initialize the Plot ---
    # Display the first frame and store the image object.
    # This object will be updated in each frame of the animation.
    im = ax.imshow(obsVec[0])

    # --- 4. Define the Update Function ---
    # This function is called for each frame of the animation.
    # The 'frame' argument is the frame number, from 0 to (frames-1).
    def update(frame):
        # Get the image data for the current frame
        new_observation = obsVec[frame]
        # Update the image data of the plot
        im.set_data(new_observation)
        return [im] # Return a list of artists that were updated

    # --- 5. Create and Save the Animation ---
    # Create the animation object
    # interval=50 means 50 milliseconds per frame (20 FPS)
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(obsVec), interval=100, blit=True)

    # Save the animation as an MP4 file
    ani.save('crafter_animation.mp4', writer='ffmpeg')

    print("Animation saved successfully as crafter_animation.mp4!")

def showObs(obs):

  observation = np.array(obs)

  height, width, _ = observation.shape

  # Define the size of each grid cell
  grid_size = 7

  # It's better to use subplots to get the 'ax' object for drawing
  fig, ax = plt.subplots(1, figsize=(5, 5))

  # Display the image first
  ax.imshow(observation)


  # Set the positions for the grid lines
  x_ticks = np.arange(0, width, grid_size)
  y_ticks = np.arange(0, height, grid_size)

  # Manually draw the grid lines
  for x in x_ticks:
      ax.axvline(x - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)

  for y in y_ticks:
      ax.axhline(y - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)


  # --- Customization and display ---

  # Clean up the plot
  ax.set_title("Observation")
  ax.axis('off') # Turn off the main axes

  # Show the final image with the grid
  plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='episode.npz')
args = parser.parse_args()


f = np.load(args.filename)
imageData = f['image']
createAnim(imageData)