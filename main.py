import PySimpleGUI as sg
import os
sg.theme("LightGreen")
        # Define the window layout
layout = [
        [sg.Text("registration form", size=(100, 1), justification="center")],
             [sg.Button("register", size=(100, 1))],
                [sg.Button("test", size=(100, 1))],
                [sg.Button("attendance", size=(100, 1))],
        [sg.Button("Exit", size=(100, 1))],
                
        ]
window = sg.Window("MENU", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "register":
        import register
    elif event == "test":
        import test 
    elif event == "attendance":
        import liveface


wiwindow.close()
    
