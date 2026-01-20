# ==============================================================================
#  COLMENA-CUÁNTICA // PROTOCOLO GÉNESIS v5.0 (GOD MODE)
#  Operador Élite: [ $USER ] // Hardware: RTX 3060 12GB
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
    echo -e "  ║  ${C_WHITE}${C_BOLD}PROTOCOLO GÉNESIS${C_RESET} ${C_YELLOW}v5.0${C_CYAN}              ║  ${C_YELLOW}${C_BOLD}NVIDIA RTX 3060 (12GB VRAM)${C_CYAN}         ║"
    echo -e "  ╚══════════════════════════════════════╩═══════════════════════════════════════╝${C_RESET}"
}

mostrar_telemetria() {
    echo -e "  ${C_BOLD}${C_BLUE}[ STATUS ]${C_RESET} ${C_CYAN}NODO:${C_RESET} ${C_WHITE}$(hostname)${C_RESET}  ${C_CYAN}DB:${C_RESET} ${C_GREEN}CONNECTED${C_RESET}  ${C_CYAN}IA:${C_RESET} ${C_YELLOW}READY${C_RESET}"
    echo -e "  ${C_BLUE}──────────────────────────────────────────────────────────────────────────────${C_RESET}"
}

mostrar_menu() {
    echo -e "  ${C_BOLD}${C_WHITE}[ SECCIONES OPERATIVAS ]${C_RESET}"
    echo ""
    echo -e "    ${C_BOLD}${C_MAGENTA}A. DESPLIEGUE & INFRAESTRUCTURA${C_RESET}"
    echo -e "      ${C_CYAN}01.${C_RESET} ${C_WHITE}INICIAR ENJAMBRE${C_RESET}     ${C_GRAY}▶ Potenciar motores 100 agentes${C_RESET}"
    echo -e "      ${C_CYAN}02.${C_RESET} ${C_WHITE}DETENER TODO${C_RESET}         ${C_GRAY}▶ Hibernación total del núcleo${C_RESET}"
    echo -e "      ${C_CYAN}03.${C_RESET} ${C_WHITE}REGENERAR${C_RESET}            ${C_GRAY}▶ Reconstruir imágenes (Full Burn)${C_RESET}"
    echo ""
    echo -e "    ${C_BOLD}${C_MAGENTA}B. INTELIGENCIA & APRENDIZAJE${C_RESET}"
    echo -e "      ${C_CYAN}04.${C_RESET} ${C_YELLOW}${C_BOLD}PRE-ENTRENAR${C_RESET}         ${C_GRAY}▶ Graduar agentes con data real${C_RESET}"
    echo -e "      ${C_CYAN}05.${C_RESET} ${C_WHITE}SESIÓN DE ESTUDIO${C_RESET}    ${C_GRAY}▶ Forzar aprendizaje horario manual${C_RESET}"
    echo ""
    echo -e "    ${C_BOLD}${C_MAGENTA}C. TELEMETRÍA & CONTROL${C_RESET}"
    echo -e "      ${C_CYAN}06.${C_RESET} ${C_YELLOW}${C_BOLD}MONITOR TESORERÍA${C_RESET}    ${C_GRAY}▶ Ver carteras en tiempo real (Consola)${C_RESET}"
    echo -e "      ${C_CYAN}07.${C_RESET} ${C_WHITE}GPU TELEMETRY${C_RESET}       ${C_GRAY}▶ Uso de VRAM y Núcleos CUDA${C_RESET}"
    echo -e "      ${C_CYAN}08.${C_RESET} ${C_WHITE}OPEN n8n${C_RESET}            ${C_GRAY}▶ Acceso al flujo de orquestación${C_RESET}"
    echo ""
    echo -e "    ${C_BOLD}${C_RED}D. PROTOCOLOS DE EMERGENCIA${C_RESET}"
    echo -e "      ${C_RED}10.${C_RESET} ${C_BOLD}GÉNESIS RESET${C_RESET}        ${C_GRAY}▶ Purga total DB y vuelta a 0.0${C_RESET}"
    echo -e "      ${C_RED}11.${C_RESET} ${C_BOLD}PURGA DOCKER${C_RESET}         ${C_GRAY}▶ Limpiar caché y liberar espacio${C_RESET}"
    echo -e "      ${C_WHITE}00.${C_RESET} ${C_BOLD}DESCONECTAR${C_RESET}          ${C_GRAY}▶ Salir de la Terminal Maestra${C_RESET}"
    echo ""
}

confirmar_accion() {
    echo -ne "\n  ${C_BOLD}${C_YELLOW}${C_BLINK}!! ADVERTENCIA !!${C_RESET} ${C_WHITE}Ejecutando protocolo crítico. ¿Proceder? [s/N]: ${C_RESET}"
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
            echo -e "\n  ${C_CYAN}[BOOT] Energizando Enjambre...${C_RESET}"
            docker compose -f docker/docker-compose.yml up -d
            echo -e "\n  ${C_GREEN}[OK] Servicios iniciados.${C_RESET}"
            read -p "  Presione Enter para volver al HUB..." ;;
        02|2)
            echo -e "\n  ${C_RED}[STOP] Hibernando sistemas...${C_RESET}"
            docker compose -f docker/docker-compose.yml down
            echo -e "\n  ${C_YELLOW}[OK] Sistemas detenidos.${C_RESET}"
            read -p "  Presione Enter para volver al HUB..." ;;
        03|3)
            if confirmar_accion; then
                echo -e "\n  ${C_YELLOW}[BURN] Reconstruyendo arquitectura nuclear...${C_RESET}"
                docker compose -f docker/docker-compose.yml build --no-cache
                echo -e "\n  ${C_GREEN}[OK] Reconstrucción finalizada.${C_RESET}"
                read -p "  Presione Enter para volver al HUB..."
            fi ;;
        04|4)
            echo -e "\n  ${C_YELLOW}╔══════════════════════════════════════════════════════════════════════════════╗"
            echo -e "  ║  ${C_WHITE}${C_BOLD}PROTOCOLO CRÓNICA: HUB DE ENTRENAMIENTO DARWINIANO${C_RESET} ${C_YELLOW}v2.0${C_CYAN}          ║"
            echo -e "  ╚══════════════════════════════════════════════════════════════════════════════╝${C_RESET}"
            echo -e "  ${C_WHITE}Flujo Sugerido:${C_RESET} ${C_CYAN}Descargar Data (1) —> Entrenar & Evolucionar (2) —> Hot-Swap (3)${C_RESET}"
            echo -e "  ${C_BLUE}──────────────────────────────────────────────────────────────────────────────${C_RESET}"
            echo -e "    ${C_CYAN}1.${C_RESET} ${C_WHITE}HARVESTER (Minería de Datos)${C_RESET}   ${C_GRAY}▶ Bajar historial de Binance para 1-100 activos${C_RESET}"
            echo -e "    ${C_CYAN}2.${C_RESET} ${C_YELLOW}${C_BOLD}EJECUTAR GIMNASIO (GPU)${C_RESET}       ${C_GRAY}▶ SAC + Selección Natural (The Reaper)${C_RESET}"
            echo -e "    ${C_CYAN}3.${C_RESET} ${C_GREEN}${C_BOLD}HOT-SWAP (Adopción)${C_RESET}           ${C_GRAY}▶ Inyectar conocimiento en el enjambre vivo${C_RESET}"
            echo -e "    ${C_CYAN}0.${C_RESET} ${C_WHITE}VOLVER AL HUB${C_RESET}"
            echo -ne "\n  ${C_BOLD}${C_MAGENTA}OPERATOR@GYM > ${C_RESET}"
            read gym_opt
            case $gym_opt in
                1)
                    echo -e "\n  ${C_CYAN}[MODULE] Harvester de Datos Masivos${C_RESET}"
                    echo -e "  ${C_GRAY}1. Inyectar Ticker Específico (ej: BTC/USDT)${C_RESET}"
                    echo -e "  ${C_GRAY}2. Descargar Top 50 Binance (USDT)${C_RESET}"
                    echo -ne "\n  ${C_BOLD}${C_MAGENTA}HARVEST > ${C_RESET}"
                    read harvest_opt
                    case $harvest_opt in
                        1)
                            echo -ne "\n  ${C_WHITE}Ingrese Ticker: ${C_RESET}"
                            read ticker
                            echo -ne "  ${C_WHITE}Días hacia atrás: ${C_RESET}"
                            read days
                            docker exec -it docker-isaac-sim-1 python -c "from src.training.data_harvester import DataHarvester; h = DataHarvester(); h.harvest_swarm_targets(['$ticker'], days_back=$days)"
                            ;;
                        2)
                            echo -ne "\n  ${C_WHITE}Días hacia atrás para cada uno: ${C_RESET}"
                            read days
                            echo -e "\n  ${C_YELLOW}[INFO] Esto puede tardar según el Rate Limit de Binance...${C_RESET}"
                            docker exec -it docker-isaac-sim-1 python -c "from src.training.data_harvester import DataHarvester; h = DataHarvester(); top = h.fetch_top_tickers(limit=50); h.harvest_swarm_targets(top, days_back=$days)"
                            ;;
                    esac
                    echo -e "\n  ${C_GREEN}[SUCCESS] Minería finalizada. Los datos están listos en el Gym.${C_RESET}"
                    ;;
                2)
                    echo -e "\n  ${C_YELLOW}[MODULE] Gimnasio GPU (Génesis Loop)${C_RESET}"
                    echo -e "  ${C_WHITE}El enjambre estudiará el portafolio y aplicará SELECCIÓN NATURAL en cada ciclo.${C_RESET}"
                    echo -ne "\n  ${C_WHITE}Cantidad de Generaciones (Ciclos de Evolución): ${C_RESET}"
                    read epochs
                    echo -e "\n  ${C_CYAN}[EXEC] Iniciando cálculo tensorial y darwiniano en RTX 3060...${C_RESET}"
                    docker exec -it docker-isaac-sim-1 env PYTHONPATH=/app python -c "from src.training.gym_engine import GymEngine; g = GymEngine(); g.train_portfolio(iterations=$epochs); g.save_brain()"
                    echo -e "\n  ${C_GREEN}╔══════════════════════════════════════════════════════════╗"
                    echo -e "  ║  ${C_BOLD}ENTRENAMIENTO & EVOLUCIÓN COMPLETADOS CON ÉXITO${C_RESET}         ║"
                    echo -e "  ╚══════════════════════════════════════════════════════════╝${C_RESET}"
                    echo -e "  ${C_WHITE}Nueva sabiduría guardada en: ${C_YELLOW}models/cerebro_colmena_entrenado.pth${C_RESET}"
                    echo -e "  ${C_CYAN}RECOMENDACIÓN: Ejecuta la Opción 3 para cargar estos pesos.${C_RESET}"
                    ;;
                3)
                    echo -e "\n  ${C_CYAN}[LINK] Sincronizando Pesos Sinápticos...${C_RESET}"
                    curl -X POST http://localhost:8000/recargar_cerebro
                    echo -e "\n  ${C_GREEN}[HOT-SWAP OK] El enjambre vivo ha evolucionado instantáneamente.${C_RESET}"
                    ;;
            esac
            read -p "  Presione Enter para volver al HUB Maestro..." ;;
        05|5)
            echo -e "\n  ${C_CYAN}[LINK] Conectando con IA Core...${C_RESET}"
            curl -X POST http://localhost:8000/aprender
            echo -e "\n  ${C_GREEN}[DONE] Sesión de estudio enviada al background.${C_RESET}"
            read -p "  Presione Enter para volver al HUB..." ;;
        06|6)
            echo -e "\n  ${C_CYAN}[MONITOR] Lanzando Auditoría de Tesorería Radical...${C_RESET}"
            python monitor_tesoro.py 
            read -p "  Monitor cerrado. Presione Enter para volver al HUB..." ;;
        07|7)
            echo -e "\n  ${C_CYAN}[GPU] Consultando telemetría CUDA (Presione Ctrl+C para salir)...${C_RESET}"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv -l 1 
            read -p "  Telemetría finalizada. Presione Enter para volver al HUB..." ;;
        08|8)
            echo -e "\n  ${C_CYAN}[N8N] Abriendo Orquestador...${C_RESET}"
            if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then start http://localhost:5678; fi 
            read -p "  Enlace enviado al navegador. Presione Enter para volver al HUB..." ;;
        10)
            if confirmar_accion; then
                echo -e "\n  ${C_RED}[RESET] Purgando base de datos de manera radical...${C_RESET}"
                docker exec -it docker-postgres-1 psql -U n8n_user -d n8n -c "TRUNCATE transactions, wallets RESTART IDENTITY;"
                echo -e "\n  ${C_GREEN}[OK] Base de datos reseteada a 0.0.${C_RESET}"
                read -p "  Sistema reseteado. Pulse Enter..."
            fi ;;
        11)
            if confirmar_accion; then
                echo -e "\n  ${C_RED}[PRUNE] Limpiando infraestructura Docker...${C_RESET}"
                docker system prune -af --volumes
                echo -e "\n  ${C_GREEN}[OK] Limpieza completada.${C_RESET}"
                read -p "  Presione Enter para volver al HUB..."
            fi ;;
        00|0)
            echo -e "\n  ${C_MAGENTA}>> Cerrando conexión. Buena caza, Operator.${C_RESET}"
            exit 0 ;;
        *)
            echo -e "\n  ${C_RED}!! ERROR: COMANDO NO RECONOCIDO !!${C_RESET}"
            sleep 1 ;;
    esac
done
