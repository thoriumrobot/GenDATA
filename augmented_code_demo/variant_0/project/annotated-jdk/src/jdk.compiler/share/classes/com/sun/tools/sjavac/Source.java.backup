/*
 * Copyright (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.sjavac;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Set;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.regex.PatternSyntaxException;

public class Source implements Comparable<Source> {

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int compareTo(Source o);

    @Override
    public int hashCode();

    public Source(Module m, String n, File f) {
    }

    public Source(Package p, String n, long lm) {
    }

    public String name();

    public String suffix();

    public Package pkg();

    public File file();

    public long lastModified();

    public void setPackage(Package p);

    public void markAsGenerated();

    public boolean isGenerated();

    public void markAsLinkedOnly();

    public boolean isLinkedOnly();

    public static Source load(Package lastPackage, String l, boolean isGenerated);

    public static void saveSources(Map<String, Source> sources, StringBuilder b);

    public static void scanRoot(File root, Set<String> suffixes, List<String> excludes, List<String> includes, Map<String, Source> foundFiles, Map<String, Module> foundModules, final Module currentModule, boolean permitSourcesWithoutPackage, boolean inGensrc, boolean inLinksrc) throws IOException, ProblemException;

    @Override
    public String toString();
}
