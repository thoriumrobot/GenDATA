/*
 * Copyright (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.taglets;

import org.checkerframework.dataflow.qual.Pure;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.TreeMap;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.util.SimpleElementVisitor14;
import javax.tools.JavaFileManager;
import javax.tools.StandardJavaFileManager;
import com.sun.source.doctree.DocTree;
import jdk.javadoc.doclet.Doclet;
import jdk.javadoc.doclet.DocletEnvironment;
import jdk.javadoc.doclet.Taglet.Location;
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
import jdk.javadoc.internal.doclets.toolkit.BaseOptions;
import jdk.javadoc.internal.doclets.toolkit.DocletElement;
import jdk.javadoc.internal.doclets.toolkit.Messages;
import jdk.javadoc.internal.doclets.toolkit.Resources;
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import static com.sun.source.doctree.DocTree.Kind.AUTHOR;
import static com.sun.source.doctree.DocTree.Kind.EXCEPTION;
import static com.sun.source.doctree.DocTree.Kind.HIDDEN;
import static com.sun.source.doctree.DocTree.Kind.LINK;
import static com.sun.source.doctree.DocTree.Kind.LINK_PLAIN;
import static com.sun.source.doctree.DocTree.Kind.PARAM;
import static com.sun.source.doctree.DocTree.Kind.PROVIDES;
import static com.sun.source.doctree.DocTree.Kind.SEE;
import static com.sun.source.doctree.DocTree.Kind.SERIAL;
import static com.sun.source.doctree.DocTree.Kind.SERIAL_DATA;
import static com.sun.source.doctree.DocTree.Kind.SERIAL_FIELD;
import static com.sun.source.doctree.DocTree.Kind.SINCE;
import static com.sun.source.doctree.DocTree.Kind.THROWS;
import static com.sun.source.doctree.DocTree.Kind.USES;
import static com.sun.source.doctree.DocTree.Kind.VERSION;
import static javax.tools.DocumentationTool.Location.TAGLET_PATH;

public class TagletManager {

    public static final char SIMPLE_TAGLET_OPT_SEPARATOR;

    public TagletManager(BaseConfiguration configuration) {
    }

    public Set<String> getAllTagletNames();

    public void initTagletPath(JavaFileManager fileManager) throws IOException;

    public void addCustomTag(String classname, JavaFileManager fileManager);

    public void loadTaglets(JavaFileManager fileManager) throws IOException;

    public void addNewSimpleCustomTag(String tagName, String header, String locations);

    void seenTag(String name);

    public void checkTags(Element element, Iterable<? extends DocTree> trees, boolean inlineTrees);

    Map<String, Taglet> getInlineTaglets();

    public List<Taglet> getSerializedFormTaglets();

    @SuppressWarnings("fallthrough")
    public List<Taglet> getBlockTaglets(Element e);

    @Pure
    public boolean isKnownCustomTag(String tagName);

    public void printReport();

    Taglet getTaglet(String name);
}
