/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Set;
    @Positive
import javax.lang.model.SourceVersion;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.util.Elements;
    @Positive
import javax.lang.model.util.Types;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject.Kind;
    @Positive
import com.sun.source.util.DocTrees;
    @Positive
import com.sun.tools.javac.code.Source;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment;

    @Positive
public class DocEnvImpl implements DocletEnvironment {

    @Positive
    public final ElementsTable etable;

    @Positive
    public final ToolEnvironment toolEnv;

    @Positive
    public DocEnvImpl(ToolEnvironment toolEnv, ElementsTable etable) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Set<? extends Element> getSpecifiedElements();

    @Positive
    @Override
    @Positive
    public Set<? extends Element> getIncludedElements();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean isIncluded(Element e);

    @Positive
    @Override
    @Positive
    public DocTrees getDocTrees();

    @Positive
    @Override
    @Positive
    public Elements getElementUtils();

    @Positive
    @Override
    @Positive
    public Types getTypeUtils();

    @Positive
    @Override
    @Positive
    public JavaFileManager getJavaFileManager();

    @Positive
    @Override
    @Positive
    public SourceVersion getSourceVersion();

    @Positive
    @Override
    @Positive
    public ModuleMode getModuleMode();

    @Positive
    @Override
    @Positive
    public Kind getFileKind(TypeElement type);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean isSelected(Element e);
    @Positive
}

// CFWR semantic augmentation - variant 1
