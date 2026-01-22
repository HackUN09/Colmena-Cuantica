#!/bin/bash
# ==============================================================================
#  COLMENA-CUÁNTICA // SISTEMA MAESTRO V1.0 (COMPLETO)
#  Operador: [ $USER ] // Hardware: NVIDIA RTX 3060 12GB
# ==============================================================================

# Colores y Efectos
C_MAGENTA='\033[38;5;201m'
C_CYAN='\033[38;5;51m'
C_YELLOW='\033[38;5;226m'
C_GREEN='\033[38;5;82m'
C_RED='\033[38;5;196m'
C_BLUE='\033[38;5;27m'
C_WHITE='\033[38;5;15m'
C_GRAY='\033[38;5;244m'
C_BOLD='\033[1m'
C_BLINK='\033[5m'
C_RESET='\033[0m'

limpiar() { clear; }

mostrar_cabecera() {
    limpiar
    echo -e "${C_CYAN}  ╔══════════════════════════════════════════════════════════════════════════════╗"
    echo -e "  ║  ${C_MAGENTA}${C_BOLD}██████╗ ██████╗ ██╗     ███╗   ███╗███████╗███╗   ██╗ █████╗  ${C_CYAN}            ║"
    echo -e "  ║  ${C_MAGENTA}${C_BOLD}██╔═══╝ ██╔══██╗██║     ████╗ ████║██╔════╝████╗  ██║██╔══██╗ ${C_CYAN}            ║"
    echo -e "  ║  ${C_MAGENTA}${C_BOLD}██║     ██║  ██║██║     ██╔████╔██║█████╗  ██╔██╗ ██║███████║ ${C_CYAN}            ║"
    echo -e "  ║  ${C_MAGENTA}${C_BOLD}██║     ██║  ██║██║     ██║╚██╔╝██║██╔══╝  ██║╚██╗██║██╔══██║ ${C_CYAN}            ║"
    echo -e "  ║  ${C_MAGENTA}${C_BOLD}╚██████╗╚██████╔╝███████╗██║ ╚═╝ ██║███████╗██║ ╚████║██║  ██║ ${C_CYAN}            ║"
    echo -e "  ╠══════════════════════════════════════╦═══════════════════════════════════════╣"
    echo -e "  ║  ${C_WHITE}${C_BOLD}SISTEMA MAESTRO${C_RESET} ${C_YELLOW}V1.0${C_CYAN}              ║  ${C_YELLOW}${C_BOLD}NVIDIA RTX 3060 (12GB VRAM)${C_CYAN}         ║"
    echo -e "  ║  ${C_GRAY}51-Dim State | Math Arsenal Completo${C_CYAN} ║  ${C_GREEN}Offline + Online Training${C_CYAN}       ║"
    echo -e "  ╚══════════════════════════════════════╩═══════════════════════════════════════╝${C_RESET}"
}

mostrar_telemetria() {
    # Check Docker status
    DOCKER_STATUS="${C_RED}STOPPED${C_RESET}"
    if docker ps > /dev/null 2>&1; then
        RUNNING=$(docker ps --format "{{.Names}}" 2>/dev/null | wc -l)
        if [ "$RUNNING" -gt 0 ]; then
            DOCKER_STATUS="${C_GREEN}RUNNING ($RUNNING)${C_RESET}"
        fi
    fi
    
    echo -e "  ${C_BOLD}${C_BLUE}[ STATUS ]${C_RESET} ${C_CYAN}NODO:${C_RESET} ${C_WHITE}$(hostname)${C_RESET}  ${C_CYAN}DOCKER:${C_RESET} ${DOCKER_STATUS}  ${C_CYAN}AGENTS:${C_RESET} ${C_YELLOW}100${C_RESET}"
    echo -e "  ${C_BLUE}──────────────────────────────────────────────────────────────────────────────${C_RESET}"
}

mostrar_menu() {
    echo -e "  ${C_BOLD}${C_WHITE}── [ A. INFRAESTRUCTURA ] ────────────────────────────────────────────────────${C_RESET}"
    echo -e "    ${C_CYAN}01.${C_RESET} ${C_WHITE}INICIAR DOCKER${C_RESET}           ${C_GRAY}▶ Docker Up (DB, n8n)${C_RESET}"
    echo -e "    ${C_CYAN}02.${C_RESET} ${C_WHITE}DETENER DOCKER${C_RESET}           ${C_GRAY}▶ Docker Down${C_RESET}"
    echo -e "    ${C_CYAN}03.${C_RESET} ${C_WHITE}VER LOGS DOCKER${C_RESET}          ${C_GRAY}▶ Logs en tiempo real${C_RESET}"
    echo -e "    ${C_CYAN}04.${C_RESET} ${C_RED}${C_BOLD}LIMPIAR DOCKER${C_RESET}           ${C_GRAY}▶ Purgar caché (docker prune)${C_RESET}"
    echo ""
    echo -e "  ${C_BOLD}${C_YELLOW}── [ B. ENTRENAMIENTO ] ──────────────────────────────────────────────────────${C_RESET}"
    echo -e "    ${C_CYAN}05.${C_RESET} ${C_YELLOW}${C_BOLD}DESCARGAR HISTÓRICOS${C_RESET}     ${C_GRAY}▶ 6 meses Top 10 (15 min)${C_RESET}"
    echo -e "    ${C_CYAN}06.${C_RESET} ${C_YELLOW}${C_BOLD}ENTRENAR OFFLINE${C_RESET}         ${C_GRAY}▶ Pre-training en GPU (2-3h)${C_RESET}"
    echo -e "    ${C_CYAN}07.${C_RESET} ${C_GREEN}${C_BOLD}INICIAR LIVE${C_RESET}             ${C_GRAY}▶ Transfer learning + n8n${C_RESET}"
    echo ""
    echo -e "  ${C_BOLD}${C_GREEN}── [ C. TESTING ] ────────────────────────────────────────────────────────────${C_RESET}"
    echo -e "    ${C_CYAN}08.${C_RESET} ${C_GREEN}${C_BOLD}TEST RÁPIDO${C_RESET}              ${C_GRAY}▶ Verificar componentes (1 min)${C_RESET}"
    echo -e "    ${C_CYAN}09.${C_RESET} ${C_GREEN}${C_BOLD}TEST COMPILACIÓN${C_RESET}         ${C_GRAY}▶ Verificar imports${C_RESET}"
    echo ""
    echo -e "  ${C_BOLD}${C_CYAN}── [ D. MONITOREO ] ──────────────────────────────────────────────────────────${C_RESET}"
    echo -e "    ${C_CYAN}10.${C_RESET} ${C_WHITE}MONITOR TESORERÍA${C_RESET}        ${C_GRAY}▶ P&L 100 agentes${C_RESET}"
    echo -e "    ${C_CYAN}11.${C_RESET} ${C_WHITE}GPU TELEMETRY${C_RESET}            ${C_GRAY}▶ VRAM + CUDA${C_RESET}"
    echo -e "    ${C_CYAN}12.${C_RESET} ${C_WHITE}LEARNING CURVES${C_RESET}          ${C_GRAY}▶ Gráficas entrenamiento${C_RESET}"
    echo -e "    ${C_CYAN}13.${C_RESET} ${C_WHITE}ABRIR n8n${C_RESET}                ${C_GRAY}▶ http://localhost:5678${C_RESET}"
    echo ""
    echo -e "  ${C_BOLD}${C_MAGENTA}── [ E. MANAGEMENT ] ─────────────────────────────────────────────────────────${C_RESET}"
    echo -e "    ${C_CYAN}14.${C_RESET} ${C_WHITE}LIMPIAR DATOS VIEJOS${C_RESET}     ${C_GRAY}▶ Borrar históricos obsoletos${C_RESET}"
    echo -e "    ${C_CYAN}15.${C_RESET} ${C_RED}${C_BOLD}RESET COMPLETO${C_RESET}           ${C_GRAY}▶ Modelos + DB + Datos${C_RESET}"
    echo -e "    ${C_CYAN}16.${C_RESET} ${C_WHITE}VER DOCUMENTACIÓN${C_RESET}        ${C_GRAY}▶ Abrir README.md${C_RESET}"
    echo ""
    echo -e "  ${C_BOLD}${C_RED}── [ F. SISTEMA ] ────────────────────────────────────────────────────────────${C_RESET}"
    echo -e "    ${C_WHITE}00.${C_RESET} ${C_BOLD}SALIR${C_RESET}                    ${C_GRAY}▶ Cerrar Sistema Maestro${C_RESET}"
    echo ""
}

confirmar_accion() {
    echo -ne "\n  ${C_BOLD}${C_YELLOW}${C_BLINK}!! ADVERTENCIA !!${C_RESET} ${C_WHITE}¿Proceder? [s/N]: ${C_RESET}"
    read confirm
    [[ $confirm == [sS] ]]
}

while true; do
    mostrar_cabecera
    mostrar_telemetria
    mostrar_menu
    echo -ne "  ${C_BOLD}${C_MAGENTA}OPERATOR@COLMENA > ${C_RESET}"
    read opt

    case $opt in
        01|1)
            echo -e "\n  ${C_CYAN}[DOCKER] Iniciando contenedores...${C_RESET}"
            if [ -f "docker/docker-compose.yml" ]; then
                docker compose -f docker/docker-compose.yml up -d
                echo -e "\n  ${C_GREEN}[OK] Contenedores iniciados${C_RESET}"
            else
                echo -e "\n  ${C_YELLOW}[INFO] No hay docker-compose.yml. Sistema funciona sin Docker.${C_RESET}"
            fi
            read -p "  Presione Enter..." ;;
        
        02|2)
            echo -e "\n  ${C_RED}[DOCKER] Deteniendo contenedores...${C_RESET}"
            if [ -f "docker/docker-compose.yml" ]; then
                docker compose -f docker/docker-compose.yml down
                echo -e "\n  ${C_YELLOW}[OK] Contenedores detenidos${C_RESET}"
            else
                echo -e "\n  ${C_YELLOW}[INFO] No hay contenedores activos.${C_RESET}"
            fi
            read -p "  Presione Enter..." ;;
        
        03|3)
            echo -e "\n  ${C_CYAN}[DOCKER] Logs en tiempo real (Ctrl+C para salir)...${C_RESET}\n"
            if [ -f "docker/docker-compose.yml" ]; then
                docker compose -f docker/docker-compose.yml logs -f
            else
                echo -e "  ${C_YELLOW}[INFO] No hay docker-compose.yml${C_RESET}"
            fi
            read -p "\n  Presione Enter..." ;;
        
        04|4)
            if confirmar_accion; then
                echo -e "\n  ${C_RED}[DOCKER] Limpiando sistema Docker...${C_RESET}"
                docker system prune -af --volumes
                echo -e "\n  ${C_GREEN}[OK] Limpieza completada${C_RESET}"
                read -p "  Presione Enter..."
            fi ;;
        
        05|5)
            echo -e "\n  ${C_YELLOW}[DOWNLOAD] Descargando datos históricos...${C_RESET}"
            echo -e "  ${C_GRAY}(Esto tomará ~10-15 minutos)${C_RESET}\n"
            python download_historical.py
            echo -e "\n  ${C_GREEN}[OK] Datos descargados en data/historical/${C_RESET}"
            read -p "  Presione Enter..." ;;
        
        06|6)
            echo -e "\n  ${C_YELLOW}[OFFLINE TRAINING] Pre-entrenamiento con datos históricos${C_RESET}"
            if [ ! -d "data/historical" ] || [ -z "$(ls -A data/historical/*.csv 2>/dev/null)" ]; then
                echo -e "  ${C_RED}[ERROR] No hay datos históricos. Ejecute opción 05 primero.${C_RESET}"
                read -p "  Presione Enter..."
            else
                echo -e "  ${C_CYAN}Configuración:${C_RESET}"
                echo -e "    - Agentes: 100"
                echo -e "    - Episodios: 500"
                echo -e "    - State: 51-dim"
                echo -e "    - Tiempo: 2-3 horas RTX 3060"
                echo -ne "\n  ${C_WHITE}¿Iniciar? [s/N]: ${C_RESET}"
                read confirm
                if [[ $confirm == [sS] ]]; then
                    echo -e "\n  ${C_CYAN}[EXEC] Iniciando train_offline_full.py...${C_RESET}\n"
                    python train_offline_full.py
                    echo -e "\n  ${C_GREEN}[OK] Entrenamiento completado${C_RESET}"
                    echo -e "  ${C_YELLOW}Modelos: models/pretrained/${C_RESET}"
                    read -p "  Presione Enter..."
                fi
            fi ;;
        
        07|7)
            echo -e "\n  ${C_GREEN}[LIVE] Iniciando entrenamiento live${C_RESET}"
            if [ -f "models/pretrained/elite_agent_1.pth" ]; then
                echo -e "  ${C_GREEN}✓ Modelos pre-entrenados detectados${C_RESET}"
            else
                echo -e "  ${C_YELLOW}⚠ Sin pre-training (recomendado: opción 06)${C_RESET}"
            fi
            echo -e "\n  ${C_CYAN}[EXEC] Iniciando main.py...${C_RESET}\n"
            python main.py
            ;;
        
        08|8)
            echo -e "\n  ${C_GREEN}[TEST] Test rápido del sistema...${C_RESET}\n"
            python tests/quick_test.py
            read -p "\n  Presione Enter..." ;;
        
        09|9)
            echo -e "\n  ${C_GREEN}[TEST] Verificando compilación...${C_RESET}\n"
            python -c "import sys; sys.path.insert(0, '.'); from scripts.train_offline_full import *; print('✅ Sistema compila correctamente')"
            echo -e "\n  ${C_GREEN}[OK] Compilación exitosa${C_RESET}"
            read -p "  Presione Enter..." ;;
        
        10)
            echo -e "\n  ${C_CYAN}[MONITOR] Monitor de tesorería...${C_RESET}\n"
            python scripts/monitor_tesoro.py
            read -p "\n  Presione Enter..." ;;
        
        11)
            echo -e "\n  ${C_CYAN}[GPU] Telemetría (Ctrl+C para salir)...${C_RESET}\n"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv -l 1
            read -p "\n  Presione Enter..." ;;
        
        12)
            echo -e "\n  ${C_CYAN}[GRAPHS] Abriendo visualizaciones...${C_RESET}"
            if [ -f "results/offline_training/training_curves.png" ]; then
                if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
                    start results/offline_training/training_curves.png
                else
                    xdg-open results/offline_training/training_curves.png
                fi
                echo -e "  ${C_GREEN}✓ Gráfica abierta${C_RESET}"
            else
                echo -e "  ${C_RED}[ERROR] No hay gráficas. Ejecute opción 06.${C_RESET}"
            fi
            read -p "  Presione Enter..." ;;
        
        13)
            echo -e "\n  ${C_CYAN}[N8N] Abriendo orquestador...${C_RESET}"
            if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
                start http://localhost:5678
            else
                xdg-open http://localhost:5678
            fi
            echo -e "  ${C_GREEN}✓ Navegador abierto: http://localhost:5678${C_RESET}"
            read -p "  Presione Enter..." ;;
        
        14)
            if confirmar_accion; then
                echo -e "\n  ${C_YELLOW}[CLEAN] Limpiando datos obsoletos...${C_RESET}"
                rm -f data/historical/*_1m.csv 2>/dev/null
                echo -e "  ${C_GREEN}✓ Archivos viejos eliminados${C_RESET}"
                read -p "  Presione Enter..."
            fi ;;
        
        15)
            if confirmar_accion; then
                echo -e "\n  ${C_RED}[RESET] Purgando sistema completo...${C_RESET}"
                rm -rf models/pretrained/* 2>/dev/null
                rm -rf results/offline_training/* 2>/dev/null
                rm -f models/cerebro_checkpoint.pth 2>/dev/null
                if [ -f "src/execution/db_maintenance.py" ]; then
                    python src/execution/db_maintenance.py
                fi
                echo -e "\n  ${C_GREEN}[OK] Sistema reiniciado${C_RESET}"
                read -p "  Presione Enter..."
            fi ;;
        
        16)
            echo -e "\n  ${C_CYAN}[DOCS] Abriendo README.md...${C_RESET}"
            if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
                start README.md
            else
                xdg-open README.md
            fi
            read -p "  Presione Enter..." ;;
        
        00|0)
            echo -e "\n  ${C_MAGENTA}>> Cerrando Sistema Maestro V1.0. Hasta pronto.${C_RESET}\n"
            exit 0 ;;
        
        *)
            echo -e "\n  ${C_RED}!! ERROR: Opción no reconocida !!${C_RESET}"
            sleep 1 ;;
    esac
done
