.PHONY: build up down sitl sim shell-sitl shell-sim x11 clean logs

build:
	docker compose build

up: x11
	docker compose up

down:
	docker compose down

sitl:
	docker compose up sitl

sim: x11
	docker compose up sim

shell-sitl:
	docker compose run --rm sitl bash

shell-sim:
	docker compose run --rm sim bash

x11:
	@xhost +local:docker > /dev/null 2>&1 || true

logs:
	docker compose logs -f

clean:
	docker compose down -v --rmi local
