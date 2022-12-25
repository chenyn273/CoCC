import ch.uzh.ifi.seal.changedistiller.ast.FileUtils;
import com.sun.corba.se.impl.resolver.SplitLocalResolverImpl;
import org.eclipse.jdt.core.dom.*;


import java.io.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class GetCCSet {
    public static boolean isValidComment(String comment) {
        int flage = 0;
        for (char c : comment.toCharArray()) {
            if (c >= 0x4E00 && c <= 0x9FA5) {
                flage = 1;
                return false;
            }
        }

        if (comment.length() < 5) {
            return false;
        }
        int count = 0;
        if (comment.contains("\n")) {
            String[] split = comment.split("\n");
            if (split.length > 20) return false;
            for (int i = 0; i < split.length; i++) {

                if (split[i].contains(".startsWith")) {
                    return false;
                }
                if (split[i].contains("===========") || split[i].contains("--------")) {
                    return false;
                }
                if (split[i].contains("public")) count++;
                if (split[i].contains("static")) count++;
                if (split[i].contains("int")) count++;
                if (split[i].contains("List")) count++;
                if (split[i].contains(">")) count++;
                if (split[i].contains("<")) count++;
                if (split[i].contains("[]")) return false;
                ;
                if (split[i].contains("=")) count++;
                if (split[i].contains("()")) count++;
                if (split[i].contains("for")) count++;
                if (split[i].contains(";")) count++;
                if (split[i].contains("double")) count++;
                if (split[i].contains("String")) count++;
                if (split[i].contains("try")) count++;
                if (split[i].contains(");")) return false;
                ;
                if (split[i].contains("(") && split[i].contains(")")) count = count + 2;
                if (split[i].contains("\"") && split[i].contains("=")) return false;
                ;
                if (split[i].contains("|")) count++;
                if (split[i].contains("(") && (split[i].contains("if") || split[i].contains("for") || split[i].contains("while")))
                    return false;
                ;
                if (split[i].contains(">") && split[i].contains("<")) return false;
                ;
                if (split[i].contains("catch")) count++;
                if (split[i].contains("Map")) count++;
                if (split[i].contains("void")) count++;
                if (split[i].contains("protected")) count++;
                if (split[i].contains("{")) count++;
                if (split[i].contains("}")) count++;
                if (split[i].contains("type info")) return false;
                ;
                if (split[i].contains("@author")) count++;
                if (split[i].contains("@version")) count++;
                if (split[i].contains(".set")) {
                    return false;
                }
                if (split[i].contains("Copyright")) {
                    return false;
                }
                if (split[i].contains("@see")) {
                    return false;
                }
                if (split[i].contains("org.")) {
                    return false;
                }
                if (split[i].contains("java.")) {
                    return false;
                }
                if (split[i].contains("util.")) {
                    return false;
                }
                if (split[i].contains("http")) {
                    return false;
                }
                if (split[i].contains("package")) count++;
                if (split[i].contains("non-Javadoc")) {
                    return false;
                }
            }
        }
        if (count >= 3) return false;


        return true;
    }

    public static boolean isValidCode(String code) {
        if (code.length() < 5) {
            return false;
        }
        if (code.contains("\n")) {
            if (code.split("\n").length > 50) {
                return false;
            }
        }
        if (code.contains("import")) {
            if (code.contains("java.")) {
                return false;
            }
            if (code.contains("package")) {
                return false;
            }
            if (code.split("import").length >= 3) {
                return false;
            }

        }

        if (code.contains("\n")) {
            int count = 0;
            for (String s : code.split("\n")) {
                if (s.contains("class")) {
                    if (s.contains("public") || s.contains("private") || s.contains("extends") || s.contains("final")) {
                        return false;
                    }
                }
                if (s.contains("/*")) count++;
                if (s.contains("*/")) count++;
            }
            if (count >= 2) return false;

        }
        return true;
    }

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
//        return 200;
    }

    public static List cmtAndCode(String filepath) throws IOException {
        int startline = 0;
        int endline = 0;
        List SetOfCCSet = new ArrayList<>();
        ASTParser astParser = ASTParser.newParser(AST.JLS3);
        BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(filepath));
        byte[] input = new byte[bufferedInputStream.available()];
        bufferedInputStream.read(input);
        bufferedInputStream.close();
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
        char[] text = new String(input).toCharArray();
        String sourcecode = "";
        for (int i = 0; i < text.length; i++) {
            sourcecode += text[i];
        }
        astParser.setSource(text);
        CompilationUnit cu = (CompilationUnit) astParser.createAST(null);
        List<Comment> commentLists = cu.getCommentList();
        List<TypeDeclaration> types = cu.types();

        if (types.size() > 0) {
            TypeDeclaration type = types.get(0);
            MethodDeclaration[] methodSet = type.getMethods();
            TypeDeclaration[] typeSet = type.getTypes();
            StringBuffer strtemp = new StringBuffer();
            StringBuffer oldComment = new StringBuffer();

            for (int i = 0; i < commentLists.size(); i++) {
                if (commentLists.get(i) instanceof Comment) {
                    int startpos = commentLists.get(i).getStartPosition();
                    int endpos = startpos + commentLists.get(i).getLength();
                    int startlineOfComment = getLineNumber(filepath, startpos);
                    int endlineOfComment = getLineNumber(filepath, endpos);

                    //BufferedReader是可以按行读取文件
                    int cnt = 0;
                    FileInputStream is = new FileInputStream(filepath);
                    BufferedReader br = new BufferedReader(new InputStreamReader(is));

                    String temp = null;
                    String startComment = null;
                    while ((temp = br.readLine()) != null) {
                        cnt++;
                        if (cnt == startlineOfComment) {
                            startComment = temp;
                            break;
                        }
                    }
                    if (!temp.trim().startsWith("/")) {
                        continue;
                    }
                    is.close();
                    br.close();

                    if (i + 1 < commentLists.size()) {
                        if (getLineNumber(filepath, commentLists.get(i).getStartPosition()) + 1 ==
                                getLineNumber(filepath, commentLists.get(i + 1).getStartPosition())) {
                            strtemp.append(sourcecode.substring(startpos, endpos) + "\n");
                            continue;
                        } else {
                            if (strtemp.equals("")) {
                                oldComment = new StringBuffer(strtemp + sourcecode.substring(startpos, endpos));
                                strtemp = new StringBuffer("");
                            } else oldComment = new StringBuffer(sourcecode.substring(startpos, endpos));
                        }
                    } else {
                        if (strtemp.equals("")) {
                            oldComment = new StringBuffer(strtemp + sourcecode.substring(startpos, endpos));
                            strtemp = new StringBuffer("");
                        } else oldComment = new StringBuffer(sourcecode.substring(startpos, endpos));

                    }


                    if (commentLists.get(i) instanceof Javadoc) {
                        startline = startlineOfComment;
                        endline = 0;
                        boolean isMethod = true;
                        int lineOfNextCommet = 99999;
                        if (i + 1 < commentLists.size()) {
                            Comment nextComment = commentLists.get(i + 1);
                            lineOfNextCommet = getLineNumber(filepath, nextComment.getStartPosition());
                        }
                        for (int j = 0; j < methodSet.length; j++) {
                            int startlineOfMethod = getLineNumber(filepath, methodSet[j].getStartPosition());
                            int methodEndPos = methodSet[j].getLength() + methodSet[j].getStartPosition();
                            int endlineOfMethod = getLineNumber(filepath, methodEndPos);
                            if (startlineOfMethod >= startline) {
                                if (startlineOfMethod > lineOfNextCommet) {
                                    endline = lineOfNextCommet - 1;
                                    isMethod = false;
                                } else {
                                    endline = endlineOfMethod;
                                    isMethod = true;
                                }
                                break;
                            }
                        }
                        List CCSet = new ArrayList<>();
                        CCSet.add(oldComment);
                        CCSet.add(startline);
                        CCSet.add(endline);
                        if (isMethod)
                            CCSet.add("METHOD_COMMENT");
                        else CCSet.add("BLOCK_COMMENT");
                        CCSet.add(endlineOfComment);
                        SetOfCCSet.add(CCSet);
                        continue;
                    }

                    for (int j = 0; j < methodSet.length; j++) {
                        int startlineOfMethod = getLineNumber(filepath, methodSet[j].getStartPosition());
                        int methodEndPos = methodSet[j].getLength() + methodSet[j].getStartPosition();
                        int endlineOfMethod = getLineNumber(filepath, methodEndPos);
                        //method类型

                        int count = 0;
                        int konghang = 0;
                        //BufferedReader是可以按行读取文件
                        FileInputStream inputStream = new FileInputStream(filepath);
                        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
                        String str = null;
                        while ((str = bufferedReader.readLine()) != null) {
                            count++;
                            if (count > endlineOfComment) {
                                if (str.equals("")) {
                                    konghang++;
                                } else break;
                            }
                        }
                        inputStream.close();
                        bufferedReader.close();

                        if (endlineOfComment + konghang + 1 == startlineOfMethod) {//202
                            startline = startlineOfComment;
                            endline = endlineOfMethod;
                            List CCSet = new ArrayList<>();
                            CCSet.add(oldComment);
                            CCSet.add(startline);
                            CCSet.add(endline);
                            CCSet.add("METHOD_COMMENT");
                            CCSet.add(endlineOfComment);
                            SetOfCCSet.add(CCSet);
                            break;
                        }//211
                        //在method中的
                        if (endlineOfComment < endlineOfMethod && endlineOfComment > startlineOfMethod) {
                            startline = startlineOfComment;
                            if (i + 1 < commentLists.size()) {
                                Comment nextComment = commentLists.get(i + 1);
                                int lineOfNextCommet = getLineNumber(filepath, nextComment.getStartPosition());
                                if (lineOfNextCommet < endlineOfMethod) {
                                    endline = lineOfNextCommet - 1;
                                    List CCSet = new ArrayList<>();
                                    CCSet.add(oldComment);
                                    CCSet.add(startline);
                                    if (endline - endlineOfComment <= 10)
                                        CCSet.add(endline);
                                    else CCSet.add(endlineOfComment + 10);
                                    CCSet.add("BLOCK_COMMENT");
                                    CCSet.add(endlineOfComment);
                                    SetOfCCSet.add(CCSet);//224
                                } else {//229
                                    endline = endlineOfMethod - 1;
                                    List CCSet = new ArrayList<>();
                                    CCSet.add(oldComment);
                                    CCSet.add(startline);
                                    if (endline - endlineOfComment <= 10)
                                        CCSet.add(endline);
                                    else CCSet.add(endlineOfComment + 10);
                                    CCSet.add("BLOCK_COMMENT");
                                    CCSet.add(endlineOfComment);
                                    SetOfCCSet.add(CCSet);
                                }
                            } else {//242
                                endline = endlineOfMethod - 1;
                                List CCSet = new ArrayList<>();
                                CCSet.add(oldComment);
                                CCSet.add(startline);
                                if (endline - endlineOfComment <= 10)
                                    CCSet.add(endline);
                                else CCSet.add(endlineOfComment + 10);
                                CCSet.add("BLOCK_COMMENT");
                                CCSet.add(endlineOfComment);
                                SetOfCCSet.add(CCSet);
                            }
                            break;
                        }
                    }//253

                    if (endline == 0) {//255
                        startline = startlineOfComment;
                        if (i + 1 < commentLists.size()) {
                            Comment nextComment = commentLists.get(i + 1);
                            int lineOfNextComment = getLineNumber(filepath, nextComment.getStartPosition());
                            endline = lineOfNextComment - 1;
                            List CCSet = new ArrayList<>();
                            CCSet.add(oldComment);
                            CCSet.add(startline);
                            if (endline - endlineOfComment <= 10)
                                CCSet.add(endline);
                            else CCSet.add(endlineOfComment + 10);
                            CCSet.add("BLOCK_COMMENT");
                            CCSet.add(endlineOfComment);
                            SetOfCCSet.add(CCSet);//267
                        } else {
                            endline = getLineNumber(filepath, cu.getStartPosition() + cu.getLength());
                            List CCSet = new ArrayList<>();
                            CCSet.add(oldComment);
                            CCSet.add(startline);
                            if (endline - endlineOfComment <= 10)
                                CCSet.add(endline);
                            else CCSet.add(endlineOfComment + 10);
                            CCSet.add("BLOCK_COMMENT");
                            CCSet.add(endlineOfComment);
                            SetOfCCSet.add(CCSet);
                        }
                    }
                }
            }
        }
        return SetOfCCSet;
    }//288

    public static List getCCSet(String filepath) throws IOException {
        List res = new ArrayList<>();

        File file = new File(filepath);
        List lists = cmtAndCode(filepath);
        int num = 0;
        for (int i = 0; i < lists.size(); i++) {

            List CCSet = (List) lists.get(i);
            String comment = ((StringBuffer) CCSet.get(0)).toString();
            int startline = (int) CCSet.get(1);
            int endline = (int) CCSet.get(2);
            String type = (String) CCSet.get(3);
            int startlineOfCode = (int) CCSet.get(4) + 1;
            //95
            FileInputStream inputStream = new FileInputStream(file);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            String str = null;
            int count = 1;
            StringBuffer code = new StringBuffer();
            while (count <= endline && (str = bufferedReader.readLine()) != null) {
                if (count >= startlineOfCode) {
                    if (str.contains("//")) {
                        String[] split = str.split("//");
                        code.append(split[0] + "\n");
                    } else code.append(str + "\n");
                }
                count++;
            }//106
            inputStream.close();
            bufferedReader.close();
            //109
            if (isValidComment(comment) && isValidCode(code.toString())) {
//                System.out.println(comment);
                List result = new ArrayList();
                result.add("oldComment:\n");
                result.add(comment + "\n");
                result.add("oldCode:\n");
                result.add(code + "\n");
                result.add("startline:" + startline + "\n");
                result.add("endline:" + endline + "\n");
                result.add("type:" + type + "\n");
                result.add("path:" + filepath + "\n");
                res.add(result);
            }

        }
        return res;

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
                if (filename.endsWith(".java") && filename.contains("/old/")) {
                    getCCSetJavaFile(filename);
                }
                //操作
            }
        } else {
        }
    }

    public static float getSimilarityRatio(String str, String target) {

        int d[][]; // 矩阵
        int n = str.length();
        int m = target.length();
        int i; // 遍历str的
        int j; // 遍历target的
        char ch1; // str的
        char ch2; // target的
        int temp; // 记录相同字符,在某个矩阵位置值的增量,不是0就是1
        if (n == 0 || m == 0) {
            return 100.0F;
        }
        d = new int[n + 1][m + 1];
        for (i = 0; i <= n; i++) { // 初始化第一列
            d[i][0] = i;
        }

        for (j = 0; j <= m; j++) { // 初始化第一行
            d[0][j] = j;
        }

        for (i = 1; i <= n; i++) { // 遍历str
            ch1 = str.charAt(i - 1);
            // 去匹配target
            for (j = 1; j <= m; j++) {
                ch2 = target.charAt(j - 1);
                if (ch1 == ch2 || ch1 == ch2 + 32 || ch1 + 32 == ch2) {
                    temp = 0;
                } else {
                    temp = 1;
                }
                // 左边+1,上边+1, 左上角+temp取最小
                d[i][j] = Math.min(Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1), d[i - 1][j - 1] + temp);
            }
        }
        return (1 - (float) d[n][m] / Math.max(str.length(), target.length())) * 100F;
    }

    public static String stringFilter(String string) {
        StringBuffer str = new StringBuffer(string);
        for (int i = 0; i < str.length(); i++) {
            if (!(str.charAt(i) >= 'A' && str.charAt(i) <= 'z')) {
                str.setCharAt(i, ' ');
            }
        }
        return str.toString();
    }

    public static void getCCSetJavaFile(String filepath) throws IOException {
        List ccSet1 = null;
        List ccSet2 = null;
        try {
            ccSet1 = getCCSet(filepath);
            ccSet2 = getCCSet(filepath.replace("/old/", "/new/"));
        } catch (Exception e) {
            return;
        }

        String oldComment = null;
        String oldCode = null;
        String oldType = null;
        String newComment = null;
        String newCode = null;
        String newType = null;
        for (int i = 0; i < ccSet1.size(); i++) {

            oldComment = (String) ((List) ccSet1.get(i)).get(1);
            oldCode = (String) ((List) ccSet1.get(i)).get(3);
            oldType = (String) ((List) ccSet1.get(i)).get(6);
            for (int j = 0; j < ccSet2.size(); j++) {
                newComment = (String) ((List) ccSet2.get(j)).get(1);
                newCode = (String) ((List) ccSet2.get(j)).get(3);
                newType = (String) ((List) ccSet2.get(j)).get(6);
                if (getSimilarityRatio(stringFilter(oldComment), stringFilter(newComment)) >= 80 &&
                        getSimilarityRatio(stringFilter(oldCode), stringFilter(newCode)) >= 80) {
                    String[] split = filepath.split("/");
                    Random rand = new Random();
                    int num1 = rand.nextInt(100);
                    int num2 = rand.nextInt(100);
                    int num3 = rand.nextInt(100);
                    FileWriter fileWriter = null;
//                    if (filepath.contains("论文")) {
//                        File file = new File("/Users/chenyn/chenyn's/研究生/DataSet/My/CCSet/" + split[split.length - 5] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                        if (!file.getParentFile().exists()) {
//                            file.getParentFile().mkdirs();
//                        }
//                        fileWriter = new FileWriter("/Users/chenyn/chenyn's/研究生/DataSet/My/CCSet/" + split[split.length - 5] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                    } else {
//                        File file = new File("/Users/chenyn/chenyn's/研究生/DataSet/My/CCSet/" + split[split.length - 4] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                        if (!file.getParentFile().exists()) {
//                            file.getParentFile().mkdirs();
//                        }
//                        fileWriter = new FileWriter("/Users/chenyn/chenyn's/研究生/DataSet/My/CCSet/" + split[split.length - 4] + "/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
//                    }
                    fileWriter = new FileWriter("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/opennms_ccset/" + num1 + num2 + num3 + split[split.length - 1].replace("/", ""));
                    fileWriter.write("==========================================CCSet==========================================\n");
                    fileWriter.write("oldComment:\n" + oldComment + "\n");
                    fileWriter.write("oldCode:\n" + oldCode + "\n");
                    fileWriter.write("newComment:\n" + newComment + "\n");
                    fileWriter.write("newCode:\n" + newCode + "\n");
                    fileWriter.write(((List) ccSet1.get(i)).get(4) + "\n");
                    fileWriter.write(((List) ccSet1.get(i)).get(5) + "\n");
                    fileWriter.write(((List) ccSet1.get(i)).get(6) + "\n");
                    fileWriter.write(((List) ccSet1.get(i)).get(7) + "\n");
                    if ((stringFilter(oldComment).replace(" ", "").equals(stringFilter(newComment).replace(" ", "")))) {
                        fileWriter.write("label:" + 0 + "\n");
                        System.out.println(0);
                    } else {
                        fileWriter.write("label:" + 1 + "\n");
                        System.out.println(1);
                    }

                    fileWriter.close();

                }
            }

        }
    }

    public static void main(String[] args) throws IOException {

        String[] projects = {

                "makagiga",
                "mc4j",
                "minuteproject",
                "mogwai",
                "nekohtml",
                "omegat"};
//        for (int i = 0; i < projects.length; i++) {
//            filesDirs(new File("/Users/chenyn/chenyn's/研究生/DataSet/CommitData/论文/" + projects[i]));
//        }

        filesDirs(new File("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/opennms_ccset"));

    }

}
