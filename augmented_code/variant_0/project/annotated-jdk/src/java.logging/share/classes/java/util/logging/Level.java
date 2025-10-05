/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.logging;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.RequiresNonNull;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.Serial;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Optional;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.function.Function;
    @Positive
import jdk.internal.loader.ClassLoaderValue;
    @Positive
import jdk.internal.access.JavaUtilResourceBundleAccess;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "interning", "nullness", "signature" })
    @Positive
@Interned
    @Positive
public class Level implements java.io.Serializable {

    @Positive
    private static final class RbAccess {
    @Positive
    }

    @Positive
    public static final Level OFF;

    @Positive
    public static final Level SEVERE;

    @Positive
    public static final Level WARNING;

    @Positive
    public static final Level INFO;

    @Positive
    public static final Level CONFIG;

    @Positive
    public static final Level FINE;

    @Positive
    public static final Level FINER;

    @Positive
    public static final Level FINEST;

    @Positive
    public static final Level ALL;

    @Positive
    @SuppressWarnings("signature")
    @Positive
    protected Level(String name, int value) {
    @Positive
    }

    @Positive
    protected Level(String name, int value, @Nullable @BinaryName String resourceBundleName) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    @BinaryName
    @Positive
    public String getResourceBundleName();

    @Positive
    public String getName();

    @Positive
    public String getLocalizedName();

    @Positive
    final String getLevelName();

    @Positive
    @Nullable
    @Positive
    final String getCachedLocalizedLevelName();

    @Positive
    @CFComment({ "nullness: This method assigns 'name' to 'localizedLevelName' in case a NullPointerException is thrown by computeLocalizedLevelName" })
    @Positive
    @SuppressWarnings({ "contracts.precondition.not.satisfied" })
    @Positive
    final synchronized String getLocalizedLevelName();

    @Positive
    @CFComment({ "nullness: level is always ensured to be non-null every time it is dereferenced" })
    @Positive
    @SuppressWarnings({ "dereference.of.nullable" })
    @Positive
    @Nullable
    @Positive
    static Level findLevel(String name);

    @Positive
    @Override
    @Positive
    public final String toString();

    @Positive
    public final int intValue();

    @Positive
    @CFComment({ "nullness: level is always ensured to be non-null every time it is dereferenced" })
    @Positive
    @SuppressWarnings({ "dereference.of.nullable" })
    @Positive
    public static synchronized Level parse(String name) throws IllegalArgumentException;

    @Positive
    @CFComment({ "nullness: It returns false in case a NullPointerException is thrown" })
    @Positive
    @SuppressWarnings({ "dereference.of.nullable" })
    @Positive
    @Override
    @Positive
    public boolean equals(@Nullable Object ox);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    static final class KnownLevel extends WeakReference<Level> {

    @Positive
        Optional<Level> mirrored();

    @Positive
        Optional<Level> referent();

    @Positive
        static synchronized void purge();

    @Positive
        static synchronized void add(Level l);

    @Positive
        static synchronized Optional<Level> findByName(String name, Function<KnownLevel, Optional<Level>> selector);

    @Positive
        static synchronized Optional<Level> findByValue(int value, Function<KnownLevel, Optional<Level>> selector);

    @Positive
        static synchronized Optional<Level> findByLocalizedLevelName(String name, Function<KnownLevel, Optional<Level>> selector);

    @Positive
        static synchronized Optional<Level> matches(Level l);
    @Positive
    }
    @Positive
}
