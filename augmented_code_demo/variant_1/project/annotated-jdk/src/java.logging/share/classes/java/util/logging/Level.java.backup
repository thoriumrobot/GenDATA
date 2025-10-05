/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.logging;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.RequiresNonNull;
import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.io.Serial;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.ResourceBundle;
import java.util.function.Function;
import jdk.internal.loader.ClassLoaderValue;
import jdk.internal.access.JavaUtilResourceBundleAccess;
import jdk.internal.access.SharedSecrets;

@AnnotatedFor({ "interning", "nullness", "signature" })
@Interned
public class Level implements java.io.Serializable {

    private static final class RbAccess {
    }

    public static final Level OFF;

    public static final Level SEVERE;

    public static final Level WARNING;

    public static final Level INFO;

    public static final Level CONFIG;

    public static final Level FINE;

    public static final Level FINER;

    public static final Level FINEST;

    public static final Level ALL;

    @SuppressWarnings("signature")
    protected Level(String name, int value) {
    }

    protected Level(String name, int value, @Nullable @BinaryName String resourceBundleName) {
    }

    @Nullable
    @BinaryName
    public String getResourceBundleName();

    public String getName();

    public String getLocalizedName();

    final String getLevelName();

    @Nullable
    final String getCachedLocalizedLevelName();

    @CFComment({ "nullness: This method assigns 'name' to 'localizedLevelName' in case a NullPointerException is thrown by computeLocalizedLevelName" })
    @SuppressWarnings({ "contracts.precondition.not.satisfied" })
    final synchronized String getLocalizedLevelName();

    @CFComment({ "nullness: level is always ensured to be non-null every time it is dereferenced" })
    @SuppressWarnings({ "dereference.of.nullable" })
    @Nullable
    static Level findLevel(String name);

    @Override
    public final String toString();

    public final int intValue();

    @CFComment({ "nullness: level is always ensured to be non-null every time it is dereferenced" })
    @SuppressWarnings({ "dereference.of.nullable" })
    public static synchronized Level parse(String name) throws IllegalArgumentException;

    @CFComment({ "nullness: It returns false in case a NullPointerException is thrown" })
    @SuppressWarnings({ "dereference.of.nullable" })
    @Override
    public boolean equals(@Nullable Object ox);

    @Override
    public int hashCode();

    static final class KnownLevel extends WeakReference<Level> {

        Optional<Level> mirrored();

        Optional<Level> referent();

        static synchronized void purge();

        static synchronized void add(Level l);

        static synchronized Optional<Level> findByName(String name, Function<KnownLevel, Optional<Level>> selector);

        static synchronized Optional<Level> findByValue(int value, Function<KnownLevel, Optional<Level>> selector);

        static synchronized Optional<Level> findByLocalizedLevelName(String name, Function<KnownLevel, Optional<Level>> selector);

        static synchronized Optional<Level> matches(Level l);
    }
}
