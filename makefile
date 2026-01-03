FILE ?= default.json

test:
	@python3 main.py test/${FILE}

test2:
	@python3 main.py test/openapi.json

clean:
	@${RM} -r schemas*

.PHONY: test test2 clean
