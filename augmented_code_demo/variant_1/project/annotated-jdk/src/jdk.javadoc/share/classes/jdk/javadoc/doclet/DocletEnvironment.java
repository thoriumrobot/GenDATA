/*
    @Positive
 * Copyright (c) 2015, 2016, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.doclet;

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
public interface DocletEnvironment {

    @Positive
    Set<? extends Element> getSpecifiedElements();

    @Positive
    Set<? extends Element> getIncludedElements();

    @Positive
    DocTrees getDocTrees();

    @Positive
    Elements getElementUtils();

    @Positive
    Types getTypeUtils();

    @Positive
    @Pure
    @Positive
    boolean isIncluded(Element e);

    @Positive
    @Pure
    @Positive
    boolean isSelected(Element e);

    @Positive
    JavaFileManager getJavaFileManager();

    @Positive
    SourceVersion getSourceVersion();

    @Positive
    ModuleMode getModuleMode();

    @Positive
    Kind getFileKind(TypeElement type);

    @Positive
    enum ModuleMode {

    @Positive
        API, ALL
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
