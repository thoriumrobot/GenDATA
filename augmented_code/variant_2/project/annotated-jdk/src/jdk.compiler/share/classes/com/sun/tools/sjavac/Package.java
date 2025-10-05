/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2012, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package com.sun.tools.sjavac;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.File;
    @Positive
import java.net.URI;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.stream.Stream;
    @Positive
import com.sun.tools.javac.util.Assert;
    @Positive
import com.sun.tools.sjavac.pubapi.PubApi;

    @Positive
public class Package implements Comparable<Package> {

    @Positive
    public Package(Module m, String n) {
    @Positive
    }

    @Positive
    public Module mod();

    @Positive
    public String name();

    @Positive
    public String dirname();

    @Positive
    public Map<String, Source> sources();

    @Positive
    public Map<String, File> artifacts();

    @Positive
    public PubApi getPubApi();

    @Positive
    public Map<String, Set<String>> typeDependencies();

    @Positive
    public Map<String, Set<String>> typeClasspathDependencies();

    @Positive
    public Set<String> dependents();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public int compareTo(Package o);

    @Positive
    public void addSource(Source s);

    @Positive
    public void parseAndAddDependency(String d, boolean cp);

    @Positive
    public void addDependency(String fullyQualifiedFrom, String fullyQualifiedTo, boolean cp);

    @Positive
    public void addDependent(String d);

    @Positive
    public boolean existsInJavacState();

    @Positive
    public boolean hasPubApiChanged(PubApi newPubApi);

    @Positive
    public void setPubapi(PubApi newPubApi);

    @Positive
    public void setDependencies(Map<String, Set<String>> ds, boolean cp);

    @Positive
    public void save(StringBuilder b);

    @Positive
    public static Package load(Module module, String l);

    @Positive
    public void saveDependencies(StringBuilder b);

    @Positive
    public void savePubapi(StringBuilder b);

    @Positive
    public static void savePackages(Map<String, Package> packages, StringBuilder b);

    @Positive
    public void addArtifact(String a);

    @Positive
    public void addArtifact(File f);

    @Positive
    public void addArtifacts(Set<URI> as);

    @Positive
    public void setArtifacts(Set<URI> as);

    @Positive
    public void loadArtifact(String l);

    @Positive
    public void saveArtifacts(StringBuilder b);

    @Positive
    public void deleteArtifacts();
    @Positive
}
