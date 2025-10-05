/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.taglets;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.PrintStream;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumMap;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.TreeMap;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.lang.model.util.SimpleElementVisitor14;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import jdk.javadoc.doclet.Doclet;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment;
    @Positive
import jdk.javadoc.doclet.Taglet.Location;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseOptions;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.DocletElement;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Messages;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Resources;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.AUTHOR;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.EXCEPTION;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.HIDDEN;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.LINK;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.LINK_PLAIN;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.PARAM;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.PROVIDES;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.SEE;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.SERIAL;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.SERIAL_DATA;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.SERIAL_FIELD;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.SINCE;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.THROWS;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.USES;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.VERSION;
    @Positive
import static javax.tools.DocumentationTool.Location.TAGLET_PATH;

    @Positive
public class TagletManager {

    @Positive
    public static final char SIMPLE_TAGLET_OPT_SEPARATOR;

    @Positive
    public TagletManager(BaseConfiguration configuration) {
    @Positive
    }

    @Positive
    public Set<String> getAllTagletNames();

    @Positive
    public void initTagletPath(JavaFileManager fileManager) throws IOException;

    @Positive
    public void addCustomTag(String classname, JavaFileManager fileManager);

    @Positive
    public void loadTaglets(JavaFileManager fileManager) throws IOException;

    @Positive
    public void addNewSimpleCustomTag(String tagName, String header, String locations);

    @Positive
    void seenTag(String name);

    @Positive
    public void checkTags(Element element, Iterable<? extends DocTree> trees, boolean inlineTrees);

    @Positive
    Map<String, Taglet> getInlineTaglets();

    @Positive
    public List<Taglet> getSerializedFormTaglets();

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public List<Taglet> getBlockTaglets(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isKnownCustomTag(String tagName);

    @Positive
    public void printReport();

    @Positive
    Taglet getTaglet(String name);
    @Positive
}
