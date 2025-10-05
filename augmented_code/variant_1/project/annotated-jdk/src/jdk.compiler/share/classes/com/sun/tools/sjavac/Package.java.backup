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
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import com.sun.tools.javac.util.Assert;
import com.sun.tools.sjavac.pubapi.PubApi;

public class Package implements Comparable<Package> {

    public Package(Module m, String n) {
    }

    public Module mod();

    public String name();

    public String dirname();

    public Map<String, Source> sources();

    public Map<String, File> artifacts();

    public PubApi getPubApi();

    public Map<String, Set<String>> typeDependencies();

    public Map<String, Set<String>> typeClasspathDependencies();

    public Set<String> dependents();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    @Override
    public int compareTo(Package o);

    public void addSource(Source s);

    public void parseAndAddDependency(String d, boolean cp);

    public void addDependency(String fullyQualifiedFrom, String fullyQualifiedTo, boolean cp);

    public void addDependent(String d);

    public boolean existsInJavacState();

    public boolean hasPubApiChanged(PubApi newPubApi);

    public void setPubapi(PubApi newPubApi);

    public void setDependencies(Map<String, Set<String>> ds, boolean cp);

    public void save(StringBuilder b);

    public static Package load(Module module, String l);

    public void saveDependencies(StringBuilder b);

    public void savePubapi(StringBuilder b);

    public static void savePackages(Map<String, Package> packages, StringBuilder b);

    public void addArtifact(String a);

    public void addArtifact(File f);

    public void addArtifacts(Set<URI> as);

    public void setArtifacts(Set<URI> as);

    public void loadArtifact(String l);

    public void saveArtifacts(StringBuilder b);

    public void deleteArtifacts();
}
