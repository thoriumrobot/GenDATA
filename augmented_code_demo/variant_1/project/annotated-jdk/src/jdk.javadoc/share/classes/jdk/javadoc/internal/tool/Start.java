/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package jdk.javadoc.internal.tool;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.text.BreakIterator;
    @Positive
import java.text.Collator;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.IllformedLocaleException;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.Collectors;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import com.sun.tools.javac.api.JavacTrees;
    @Positive
import com.sun.tools.javac.file.BaseFileManager;
    @Positive
import com.sun.tools.javac.file.JavacFileManager;
    @Positive
import com.sun.tools.javac.jvm.Target;
    @Positive
import com.sun.tools.javac.main.Arguments;
    @Positive
import com.sun.tools.javac.main.CommandLine;
    @Positive
import com.sun.tools.javac.util.ClientCodeException;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.Log;
    @Positive
import com.sun.tools.javac.util.StringUtils;
    @Positive
import jdk.javadoc.doclet.Doclet;
    @Positive
import jdk.javadoc.doclet.Doclet.Option;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment;
    @Positive
import jdk.javadoc.doclet.StandardDoclet;
    @Positive
import jdk.javadoc.internal.Versions;
    @Positive
import jdk.javadoc.internal.tool.Main.Result;
    @Positive
import jdk.javadoc.internal.tool.ToolOptions.ToolOption;
    @Positive
import static javax.tools.DocumentationTool.Location.*;
    @Positive
import static jdk.javadoc.internal.tool.Main.Result.*;

    @Positive
public class Start {

    @Positive
    public Start(Context context) {
    @Positive
    }

    @Positive
    void showOption(List<String> names, String parameters, String description);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    Result begin(String... argv);

    @Positive
    public boolean begin(Class<?> docletClass, Iterable<String> options, Iterable<? extends JavaFileObject> fileObjects);

    @Positive
    boolean matches(List<String> names, String arg);

    @Positive
    boolean matches(Doclet.Option option, String arg);

    @Positive
    int consumeDocletOption(int idx, List<String> args, boolean isToolOption) throws OptionException;

    @Positive
    void error(String key, Object... args);
    @Positive
}

// CFWR semantic augmentation - variant 1
