def sendSMS(Tpot,phone_to):
	'''
	send Tpot via SMS.
	Syntax:
	-------
	sendSMS(Tpot,phone_to)
	Parameters:
	-----------
	Tpot: float, pot temperature.
	phone_to: str, phone number to with country code in front, e.g. +1352*******.
	'''
	# we import the Twilio client from the dependency we just installed
	from twilio.rest import Client

	# the following line needs your Twilio Account SID and Auth Token
	client = Client("ACc087d2949d11e86bfe0c50d333f53218", "77a1f0fd87293cda99c70f6759842576")

	# change the "from_" number to your Twilio number and the "to" number
	# to the phone number you signed up for Twilio with, or upgrade your
	# account to send SMS to any phone number
	client.messages.create(to=phone_to, 
			       from_="+13192505950", 
			       body="Tpot=%.3f K"%Tpot)

