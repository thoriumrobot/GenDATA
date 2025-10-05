/*
 * Copyright (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.module;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.InputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import static jdk.internal.module.Checks.*;
import static java.util.Objects.*;
import jdk.internal.module.Checks;
import jdk.internal.module.ModuleInfo;

public class ModuleDescriptor implements Comparable<ModuleDescriptor> {

    public enum Modifier {

        OPEN, AUTOMATIC, SYNTHETIC, MANDATED
    }

    public static final class Requires implements Comparable<Requires> {

        public enum Modifier {

            TRANSITIVE, STATIC, SYNTHETIC, MANDATED
        }

        public Set<Modifier> modifiers();

        public String name();

        public Optional<Version> compiledVersion();

        public Optional<String> rawCompiledVersion();

        @Override
        public int compareTo(Requires that);

        @Override
        public boolean equals(Object ob);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    public static final class Exports implements Comparable<Exports> {

        public enum Modifier {

            SYNTHETIC, MANDATED
        }

        public Set<Modifier> modifiers();

        public boolean isQualified();

        public String source();

        public Set<String> targets();

        @Override
        public int compareTo(Exports that);

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object ob);

        @Override
        public String toString();
    }

    public static final class Opens implements Comparable<Opens> {

        public enum Modifier {

            SYNTHETIC, MANDATED
        }

        public Set<Modifier> modifiers();

        public boolean isQualified();

        public String source();

        public Set<String> targets();

        @Override
        public int compareTo(Opens that);

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object ob);

        @Override
        public String toString();
    }

    public static final class Provides implements Comparable<Provides> {

        public String service();

        public List<String> providers();

        public int compareTo(Provides that);

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object ob);

        @Override
        public String toString();
    }

    public static final class Version implements Comparable<Version> {

        public static Version parse(String v);

        @Override
        public int compareTo(Version that);

        @Override
        public boolean equals(Object ob);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    public String name();

    public Set<Modifier> modifiers();

    public boolean isOpen();

    public boolean isAutomatic();

    public Set<Requires> requires();

    public Set<Exports> exports();

    public Set<Opens> opens();

    public Set<String> uses();

    public Set<Provides> provides();

    public Optional<Version> version();

    public Optional<String> rawVersion();

    public String toNameAndVersion();

    public Optional<String> mainClass();

    public Set<String> packages();

    public static final class Builder {

        Set<String> packages();

        public Builder requires(Requires req);

        public Builder requires(Set<Requires.Modifier> ms, String mn, Version compiledVersion);

        Builder requires(Set<Requires.Modifier> ms, String mn, String rawCompiledVersion);

        public Builder requires(Set<Requires.Modifier> ms, String mn);

        public Builder requires(String mn);

        public Builder exports(Exports e);

        public Builder exports(Set<Exports.Modifier> ms, String pn, Set<String> targets);

        public Builder exports(Set<Exports.Modifier> ms, String pn);

        public Builder exports(String pn, Set<String> targets);

        public Builder exports(String pn);

        public Builder opens(Opens obj);

        public Builder opens(Set<Opens.Modifier> ms, String pn, Set<String> targets);

        public Builder opens(Set<Opens.Modifier> ms, String pn);

        public Builder opens(String pn, Set<String> targets);

        public Builder opens(String pn);

        public Builder uses(String service);

        public Builder provides(Provides p);

        public Builder provides(String service, List<String> providers);

        public Builder packages(Set<String> pns);

        public Builder version(Version v);

        public Builder version(String vs);

        public Builder mainClass(String mc);

        public ModuleDescriptor build();
    }

    @Override
    public int compareTo(ModuleDescriptor that);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object ob);

    @Override
    public int hashCode();

    @Override
    public String toString();

    public static Builder newModule(String name, Set<Modifier> ms);

    public static Builder newModule(String name);

    public static Builder newOpenModule(String name);

    public static Builder newAutomaticModule(String name);

    public static ModuleDescriptor read(InputStream in, Supplier<Set<String>> packageFinder) throws IOException;

    public static ModuleDescriptor read(InputStream in) throws IOException;

    public static ModuleDescriptor read(ByteBuffer bb, Supplier<Set<String>> packageFinder);

    public static ModuleDescriptor read(ByteBuffer bb);
}
