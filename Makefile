help:
	@echo "HELP"
	@echo "----"
	@echo "* Start App - Console *"
	@echo "Type \"make start_app_console\" to   start the app in console mode."
	@echo "* Start App - WebApp *"
	@echo "Type \"make start_app_webapp\" to   start the app as a WebApp made using Dash."
	@echo "* Start App - File Evaluation *"
	@echo "Type \"make start_app_file_eval\" to   start the app and test a text file."

start_app_console:
	python -m application.app --mode=console

start_app_webapp:
	python -m application.app --mode=webapp

start_app_file_eval:
	python -m application.app --mode=file_eval
