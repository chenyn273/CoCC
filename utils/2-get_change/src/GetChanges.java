import ch.uzh.ifi.seal.changedistiller.ChangeDistiller;
import ch.uzh.ifi.seal.changedistiller.ast.FileUtils;
import ch.uzh.ifi.seal.changedistiller.distilling.FileDistiller;
import ch.uzh.ifi.seal.changedistiller.model.entities.Insert;
import ch.uzh.ifi.seal.changedistiller.model.entities.SourceCodeChange;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GetChanges {
    private static int count = 0;
    private static String now = "";

    public static int getLineNumber(String file, int position) {
        int lineNum = 1;
        String fileContent = FileUtils.getContent(new File(file));
        char[] charArray = fileContent.toCharArray();
        for (int i = 0; i < position && i < charArray.length; i++) {
            if (charArray[i] == '\n') {
                lineNum++;
            }
        }
        return lineNum;
    }

    public static void filesDirs(File file1) throws IOException {
        if (file1 != null) {
            //第二层路径不为空，判断是文件夹还是文件
            if (file1.isDirectory()) {
                //进入这里说明为文件夹，此时需要获得当前文件夹下所有文件，包括目录
                File[] files = file1.listFiles();
                for (File flies2 : files) {
                    filesDirs(flies2);
                }
            } else {
                String filename = file1.toString();
                if (filename.endsWith(".java")) {
                    System.out.println(now);
                    System.out.println(count++);
                    try {
                        List ccSetWithChange = getCCSetWithChange(filename);
                        if (ccSetWithChange != null) {
                            String changes = (String) ccSetWithChange.get(0);
                            String path = (String) ccSetWithChange.get(1);
                            String[] split = path.split("/");
                            if (path.contains("/论文/")) {
                                if (!now.equals(split[split.length - 5])) {
                                    now = split[split.length - 5];
                                    count = 0;
                                }
                            } else {
                                if (!now.equals(split[split.length - 4])) {
                                    now = split[split.length - 4];
                                    count = 0;
                                }
                            }
                            Random rand = new Random();
                            int num1 = rand.nextInt(100);
                            int num2 = rand.nextInt(100);
                            int num3 = rand.nextInt(100);
                            FileWriter fileWriter;
//                            if (path.contains("/论文/")) {
//                                File file = new File("/Users/chenyn/chenyn's/研究生/DataSet/My/Changes/" + split[split.length - 5] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                                if (!file.getParentFile().exists()) {
//                                    file.getParentFile().mkdirs();
//                                }
//                                fileWriter = new FileWriter("/Users/chenyn/chenyn's/研究生/DataSet/My/Changes/" + split[split.length - 5] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                            } else {
//                                File file = new File("/Users/chenyn/chenyn's/研究生/DataSet/My/Changes/" + split[split.length - 4] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                                if (!file.getParentFile().exists()) {
//                                    file.getParentFile().mkdirs();
//                                }
//                                fileWriter = new FileWriter("/Users/chenyn/chenyn's/研究生/DataSet/My/Changes/" + split[split.length - 4] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                            }
                            fileWriter = new FileWriter("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/freecol_change/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));

                            fileWriter.write(changes);
                            fileWriter.close();
                        }
                    } catch (Exception e) {

                    }
                }
            }
        } else {
            System.out.println("文件不存在");
        }
    }

    public static List getCCSetWithChange(String filepath) throws IOException {
        List res = new ArrayList<>();
        FileInputStream inputStream = new FileInputStream(filepath);
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        StringBuffer result = new StringBuffer();
        String str = null;
        String path = null;
        int startline = 0;
        int endline = 0;
        while ((str = bufferedReader.readLine()) != null) {
            result.append(str + "\n");
            if (str.startsWith("startline:")) startline = Integer.parseInt(str.split(":")[1]);
            if (str.startsWith("endline:")) endline = Integer.parseInt(str.split(":")[1]);
            if (str.startsWith("path:")) path = str.split(":")[1];
        }
        inputStream.close();
        bufferedReader.close();

        String change = getChange(path, startline, endline);
        if (change.startsWith("changeNum:0")) return null;
        result.append("\n=========================================Changes=========================================\n");
        result.append(change);
        res.add(result.toString());
        res.add(path);
        return res;
    }

    public static String getChange(String filepath, int startline, int endline) {
        File file1 = new File(filepath);
        File file2 = new File(filepath.replace("/old/", "/new/"));
        int change_num = 0;
        StringBuffer result = new StringBuffer();
        FileDistiller distiller = ChangeDistiller.createFileDistiller(ChangeDistiller.Language.JAVA);
        try {
            distiller.extractClassifiedSourceCodeChanges(file1, file2);
        } catch (Exception e) {
            System.err.println("Warning: error while change distilling. " + e.getMessage());
        }
        List<SourceCodeChange> changes = distiller.getSourceCodeChanges();
        if (changes != null) {
            //对change进行细粒度操作
            //记录start位置 和 类型
            for (SourceCodeChange change : changes) {
                if (change instanceof Insert || change.getChangedEntity().getType().toString().contains("COMMENT") || change.getChangedEntity().getType().toString().contains("DOC")) {
                    continue;
                }
                int changeStartLine = getLineNumber(filepath, change.getChangedEntity().getStartPosition());
                int changeEndLine = getLineNumber(filepath, change.getChangedEntity().getEndPosition());
                if (changeStartLine >= startline && changeEndLine <= endline) {
                    result.append("change " + change_num + " : " + changeStartLine + "," + changeEndLine + "\n");
                    result.append("change:" + change.toString() + "\n");
                    result.append("change_getClass:" + change.getClass().toString() + "\n");
                    result.append("change_type:" + change.getChangeType().toString() + "\n");
                    result.append("change_entity:" + change.getChangedEntity().toString() + "\n");
                    result.append("change_entity_SourceRange:" + change.getChangedEntity().getSourceRange().toString() + "\n");
                    result.append("change_entity_uniqueName:" + change.getChangedEntity().getUniqueName() + "\n");
                    result.append("change_entity_getClass:" + change.getChangedEntity().getClass().toString() + "\n");
                    result.append("change_entity_label:" + change.getChangedEntity().getLabel() + "\n");
                    result.append("change_entity_type:" + change.getChangedEntity().getType().toString() + "\n");
                    result.append("isNative:" + String.valueOf(change.getChangedEntity().isNative()) + "\n");
                    result.append("isPrivate:" + String.valueOf(change.getChangedEntity().isPrivate()) + "\n");
                    result.append("isVolatile:" + String.valueOf(change.getChangedEntity().isVolatile()) + "\n");
                    result.append("isAbstract:" + String.valueOf(change.getChangedEntity().isAbstract()) + "\n");
                    result.append("isFinal:" + String.valueOf(change.getChangedEntity().isFinal()) + "\n");
                    result.append("isProtected:" + String.valueOf(change.getChangedEntity().isProtected()) + "\n");
                    result.append("isPublic:" + String.valueOf(change.getChangedEntity().isPublic()) + "\n");
                    result.append("isStatic:" + String.valueOf(change.getChangedEntity().isStatic()) + "\n");
                    result.append("isSynchronized:" + String.valueOf(change.getChangedEntity().isSynchronized()) + "\n");
                    result.append("isTransient:" + String.valueOf(change.getChangedEntity().isTransient()) + "\n");
                    result.append("isBodyChange:" + String.valueOf(change.getChangeType().isBodyChange()) + "\n");
                    result.append("isDeclarationChange:" + String.valueOf(change.getChangeType().isDeclarationChange()) + "\n");
                    result.append("\n");
                    change_num++;
                }

            }
        }
        System.out.println(change_num);
        result.append("changeNum:" + change_num + "\n");
        return result.toString();
    }

    public static void main(String[] args) throws IOException {
//        String[] proj = {
//
//                "neuroph",
//                "opennms",
//                "symmetricds",
//                "ted",
//                "timeslottracker",
//                "toxtree",
//                "tudu",
//                "tvschedulerpro",
//                "txm",
//                "ucanaccess",
//                "dcm4che",
//                "docfetcher",
//                "documentburster",
//                "ejbca",
//                "freecol",
//                "healpix",
//                "hieos",
//                "hsqldb",
//                "htmlunit",
//                "isphere",
//                "jamwiki",
//                "jedit",
//                "jhotdraw",
//                "jmstoolbox",
//                "joda-time",
//                "joeq",
//                "jquant",
//                "jsquadleader",
//                "jtds",
//                "jump-pilot",
//                "jython",
//                "kablink",
//                "makagiga",
//                "mc4j",
//                "minuteproject",
//                "mogwai",
//                "nekohtml",
//                "omegat"
//
//
//        };
//        for (int i = 0; i < proj.length; i++) {
//            filesDirs(new File("/Users/chenyn/chenyn's/研究生/DataSet/My/CCSet/" + proj[i]));
//        }
        filesDirs(new File("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/freecol_ccset"));
    }

}
