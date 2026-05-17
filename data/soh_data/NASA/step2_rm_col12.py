import csv
import os
import argparse
from pathlib import Path

def process_csv_files(input_folder, output_folder):
    """
    处理文件夹中的所有CSV文件：删除前两列并保存到输出文件夹
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"在文件夹 {input_folder} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件需要处理...")
    
    processed_count = 0
    for filename in csv_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
                 open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                for row in reader:
                    # 删除前两列（保留第三列及之后的所有列）
                    if len(row) > 2:
                        writer.writerow(row[2:])
                    elif row:  # 处理行不足三列的情况
                        # 如果只有一列，删除后变为空行；如果有两列，删除后保留空行
                        writer.writerow([])
            
            processed_count += 1
            print(f"已处理: {filename}")
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    print(f"\n处理完成! 已处理 {processed_count}/{len(csv_files)} 个文件")
    print(f"结果保存在: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量处理文件夹中的CSV文件：删除前两列')
    parser.add_argument('input_folder', help='包含CSV文件的输入文件夹路径')
    parser.add_argument('output_folder', nargs='?', default='output_removed_columns', 
                        help='输出文件夹路径（可选，默认为"output_removed_columns"）')
    
    args = parser.parse_args()
    
    process_csv_files(args.input_folder, args.output_folder)