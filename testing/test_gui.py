import pybullet as p

try:
    # Försök att ansluta till PyBullet GUI
    p.connect(p.GUI)
    print("PyBullet GUI-anslutning lyckades! Det verkar inte finnas någon aktiv GUI-session.")
    p.disconnect()  # Koppla från om anslutningen lyckas
except Exception as e:
    # Om det misslyckas skrivs ett felmeddelande ut
    print(f"Misslyckades att ansluta till PyBullet GUI: {e}")

