#! -*- coding: utf-8 -*-
"""
Basic setup to talk to the VNA
"""
import pyvisa


# Fixed address for the ECEE lab
instr_addr = 'TCPIP0::K-N5222B-21927::hislip0,4880::INSTR'

# set up the PNA to measure s21 for the specific instrument GPIB0::16::INSTR
rm = pyvisa.ResourceManager()

# handle failure to open the GPIB resource #this is an issue when connecting
# to the PNA-X from newyork rather than ontario
try:
    pna = rm.open_resource(instr_addr)

    ## Attempt to fix the timeout error in averaging command
    pna.timeout = None
except Exception as ex:
    print(f'\n----------\nException:\n{ex}\n----------\n')
    raise RuntimeError('Bad instrument address.')

# Reading and writing commands
gpoints = pna.query(f'SENSe1:SWEep:POINts?')
print(f'gpoints: {gpoints}')
pna.write('INITiate:CONTinuous ON')
pna.write('OUTPut:STATe ON')
pna.write('FORMat ASCII')
pna.write('OUTPut:STATe OFF')
