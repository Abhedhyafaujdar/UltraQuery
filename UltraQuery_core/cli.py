import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from UltraQuery_core import internal
    from UltraQuery_core.ai_tools import AIClient, AIClientError, WebCmdClient, WebCmdError, advanced_payload, csv_context, tool_payload
else:
    from . import internal
    from .ai_tools import AIClient, AIClientError, WebCmdClient, WebCmdError, advanced_payload, csv_context, tool_payload

def main():

    parser=argparse.ArgumentParser(description="Welcome to UltraQuery CLI Tool")
    parser.add_argument("-f","--file",help="type -f to enter the file")
    parser.add_argument("-v","--view",action="store_true",help="type -v to view the dataframe")
    parser.add_argument("-col","--columns",help="type -col to view the columns list")
    parser.add_argument("-vc","--column_data")
    parser.add_argument("-plt","--plot",action="store_true",help="Plot the graphs")
    parser.add_argument("-typ","--type",help="Available type of graphs :\n['bar','pie','line','scatter','histogram']")
    parser.add_argument("-x","--xAxis",help="Give the X axis here")
    parser.add_argument("-y","--yAxis",help="give the y axis here" )
    parser.add_argument("-df","--DataFrame",help="Use -df for building dataframes")
    parser.add_argument("-l","--limit",type=int,help="type -l to limit the number of rows")
    parser.add_argument("--context", help="Return compact CSV context as JSON")
    parser.add_argument("--web-search", help="Research a topic through WebCmd OmniSearch")
    parser.add_argument("--with-web", action="store_true", help="Include WebCmd research in ask")
    parser.add_argument("--auto-plot", action="store_true", help="Choose plot columns automatically")
    parser.add_argument("--compare", action="store_true", help="Compare multiple numeric columns")
    parser.add_argument("--scrape-urls", action="store_true", help="Fetch public URLs found in CSV files")
    parser.add_argument("--max-urls", type=int, default=5, help="Maximum URLs to fetch per analysis")
    parser.add_argument("--question", help="Question for the analyze command")
    parser.add_argument("--payload-only", action="store_true", help="Print analyze context without calling an AI model")
    parser.add_argument("--dashboard", action="store_true", help="Create a futuristic smart matplotlib dashboard")
    parser.add_argument("--output", default="ultraquery_dashboard.png", help="Dashboard output image path")
    parser.add_argument("--open", action="store_true", help="Open the generated dashboard after saving")
    parser.add_argument("command", nargs="?", help="Use 'ask' for an AI answer")
    parser.add_argument("command_args", nargs="*", help="Arguments for the selected command")
    args=parser.parse_args()

    if args.context:
        try:
            print(json.dumps(csv_context(args.context), indent=2, default=str))
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        return

    if args.web_search:
        try:
            print(json.dumps(WebCmdClient().search(args.web_search), indent=2, default=str))
        except WebCmdError as error:
            parser.error(str(error))
        return

    if args.command == "ask":
        if len(args.command_args) < 2:
            parser.error("ask requires a CSV path and a question")
        csv_path = args.command_args[0]
        question = " ".join(args.command_args[1:])
        try:
            context = tool_payload(csv_path, question, WebCmdClient() if args.with_web else None)
            print(AIClient().generate_answer(question, context))
        except (AIClientError, WebCmdError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        return

    if args.command == "analyze":
        if not args.command_args:
            parser.error("analyze requires one or more CSV paths")
        question = args.question or "Compare these datasets and explain the most important findings."
        webcmd = WebCmdClient() if (args.with_web or args.scrape_urls) else None
        try:
            context = advanced_payload(
                args.command_args,
                question,
                webcmd=webcmd,
                scrape_urls=args.scrape_urls,
                max_urls=args.max_urls,
            )
            if args.payload_only:
                print(json.dumps(context, indent=2, default=str))
            else:
                print(AIClient().generate_answer(question, context))
        except (AIClientError, WebCmdError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        return

    if args.command == "multi-plot":
        if len(args.command_args) < 2:
            parser.error("multi-plot requires at least two CSV paths")
        try:
            internal.UltraQuery().multi_dataset_plot(args.command_args, args.yAxis, args.output if args.output != "ultraquery_dashboard.png" else None)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        return

    if args.dashboard:
        if not args.file:
            parser.error("--dashboard requires --file PATH")
        try:
            context = None
            pages = []
            summary = None
            if args.scrape_urls or args.with_web:
                context = advanced_payload(
                    [args.file],
                    args.question or "Summarize the most important findings in this dataset.",
                    webcmd=WebCmdClient(),
                    scrape_urls=args.scrape_urls,
                    max_urls=args.max_urls,
                )
                pages = context.get("scraped_pages", [{}])[0].get("pages", []) if args.scrape_urls else []
                if args.with_web:
                    summary = AIClient().generate_answer(
                        args.question or "Summarize the most important findings in this dataset.",
                        context,
                    )
            internal.UltraQuery().smart_dashboard(args.file, pages, summary, args.output)
            if args.open:
                output_path = Path(args.output).resolve()
                if not output_path.exists():
                    parser.error(f"Dashboard was not created: {output_path}")
                if sys.platform == "win32":
                    import os
                    os.startfile(str(output_path))
                else:
                    parser.error("--open is currently supported on Windows only")
        except (AIClientError, WebCmdError, FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        return
    
    
    if args.view:
        internal.UltraQuery().viewfile(args.file,args.limit)
    
    if args.columns:
        internal.UltraQuery().viewcolumn(args.file)
    
    if args.column_data:
        internal.UltraQuery().viewdata(args.file,args.limit)

    if args.DataFrame:
        internal.UltraQuery().dataframe(args.file,args.limit)
    
    if args.plot:
        internal.UltraQuery().plot(args.file,args.xAxis,args.yAxis,args.type)

    if args.auto_plot:
        internal.UltraQuery().auto_plot(args.file, args.type or "auto")

    if args.compare:
        internal.UltraQuery().comparison_plot(args.file, args.xAxis, args.yAxis)


if __name__ == "__main__":
    main()

