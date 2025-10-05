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
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import com.sun.tools.sjavac.pubapi.PubApi;

public class Module implements Comparable<Module> {

    public Module(String n, String dn) {
    }

    public String name();

    public String dirname();

    public Map<String, Package> packages();

    public Map<String, Source> sources();

    public Map<String, File> artifacts();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    @Override
    public int compareTo(Module o);

    public void save(StringBuilder b);

    public static Module load(String l);

    public static void saveModules(Map<String, Module> ms, StringBuilder b);

    public void addPackage(Package p);

    public Package lookupPackage(String pkg);

    public void addSource(String pkg, Source src);

    public Source lookupSource(String path);

    public void addArtifacts(String pkg, Set<URI> as);

    public void setDependencies(String pkg, Map<String, Set<String>> deps, boolean cp);

    public void setPubapi(String pkg, PubApi ps);

    public boolean hasPubapiChanged(String pkg, PubApi newPubApi);
}
