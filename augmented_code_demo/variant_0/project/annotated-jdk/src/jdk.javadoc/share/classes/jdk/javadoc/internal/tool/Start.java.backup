/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package jdk.javadoc.internal.tool;

import org.checkerframework.dataflow.qual.Pure;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.BreakIterator;
import java.text.Collator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.IllformedLocaleException;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import com.sun.tools.javac.api.JavacTrees;
import com.sun.tools.javac.file.BaseFileManager;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.jvm.Target;
import com.sun.tools.javac.main.Arguments;
import com.sun.tools.javac.main.CommandLine;
import com.sun.tools.javac.util.ClientCodeException;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.StringUtils;
import jdk.javadoc.doclet.Doclet;
import jdk.javadoc.doclet.Doclet.Option;
import jdk.javadoc.doclet.DocletEnvironment;
import jdk.javadoc.doclet.StandardDoclet;
import jdk.javadoc.internal.Versions;
import jdk.javadoc.internal.tool.Main.Result;
import jdk.javadoc.internal.tool.ToolOptions.ToolOption;
import static javax.tools.DocumentationTool.Location.*;
import static jdk.javadoc.internal.tool.Main.Result.*;

public class Start {

    public Start(Context context) {
    }

    void showOption(List<String> names, String parameters, String description);

    @SuppressWarnings("deprecation")
    Result begin(String... argv);

    public boolean begin(Class<?> docletClass, Iterable<String> options, Iterable<? extends JavaFileObject> fileObjects);

    boolean matches(List<String> names, String arg);

    boolean matches(Doclet.Option option, String arg);

    int consumeDocletOption(int idx, List<String> args, boolean isToolOption) throws OptionException;

    void error(String key, Object... args);
}
