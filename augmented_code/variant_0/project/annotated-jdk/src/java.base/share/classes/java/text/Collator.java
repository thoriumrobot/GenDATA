/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

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
import java.lang.ref.SoftReference;
    @Positive
import java.text.spi.CollatorProvider;
    @Positive
import java.util.Locale;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;

    @Positive
public abstract class Collator implements java.util.Comparator<Object>, Cloneable {

    @Positive
    public static final int PRIMARY;

    @Positive
    public static final int SECONDARY;

    @Positive
    public static final int TERTIARY;

    @Positive
    public static final int IDENTICAL;

    @Positive
    public static final int NO_DECOMPOSITION;

    @Positive
    public static final int CANONICAL_DECOMPOSITION;

    @Positive
    public static final int FULL_DECOMPOSITION;

    @Positive
    public static synchronized Collator getInstance();

    @Positive
    public static Collator getInstance(Locale desiredLocale);

    @Positive
    public abstract int compare(String source, String target);

    @Positive
    @Override
    @Positive
    public int compare(Object o1, Object o2);

    @Positive
    public abstract CollationKey getCollationKey(String source);

    @Positive
    public boolean equals(String source, String target);

    @Positive
    public synchronized int getStrength();

    @Positive
    public synchronized void setStrength(int newStrength);

    @Positive
    public synchronized int getDecomposition();

    @Positive
    public synchronized void setDecomposition(int decompositionMode);

    @Positive
    public static synchronized Locale[] getAvailableLocales();

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object that);

    @Positive
    @Override
    @Positive
    public abstract int hashCode();

    @Positive
    protected Collator() {
    @Positive
    }
    @Positive
}
