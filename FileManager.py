from xml.dom import minidom
import csv


class FileManager:

    @staticmethod
    def get_messages_from_xml(path):
        xml_file = minidom.parse(path)
        txt = xml_file.getElementsByTagName("text")

        msgs = [x.firstChild.data for x in txt]
        return msgs

    @staticmethod
    def get_messages_from_csv(path):
        with open(path, 'r') as csv_file:
            return csv.reader(csv_file, delimiter='\n')

    @staticmethod
    def get_messages_from_txt(path):
        with open(path, 'r') as txt_file:
            return [line[:-1] for line in txt_file]

    @staticmethod
    def find_adjacent_duplicities(strings):
        ans = []
        i = 0
        for i in range(strings.__len__()-1):
            if strings[i] == strings[i+1]:
                ans.append(i)
        return ans

    @staticmethod
    def append_to_file(path, txt):
        with open(path, 'a+') as myfile:
            myfile.write(txt + '\n')


    @staticmethod
    def overwrite_to_file(path, txt):
        with open(path, 'w') as myfile:
            myfile.write(txt + '\n')

    @classmethod
    def remove_adjacent_duplicities(cls, strings):
        to_delete = cls.find_adjacent_duplicities(strings)
        for i in sorted(to_delete, reverse=True):
            try:
                del strings[i]
            except Exception:
                print(i, 'bad')
                print(i, ':', strings[i])
