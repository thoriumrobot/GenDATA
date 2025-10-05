/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.ref.SoftReference;
import java.text.spi.CollatorProvider;
import java.util.Locale;
import java.util.ResourceBundle;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleServiceProviderPool;

public abstract class Collator implements java.util.Comparator<Object>, Cloneable {

    public static final int PRIMARY;

    public static final int SECONDARY;

    public static final int TERTIARY;

    public static final int IDENTICAL;

    public static final int NO_DECOMPOSITION;

    public static final int CANONICAL_DECOMPOSITION;

    public static final int FULL_DECOMPOSITION;

    public static synchronized Collator getInstance();

    public static Collator getInstance(Locale desiredLocale);

    public abstract int compare(String source, String target);

    @Override
    public int compare(Object o1, Object o2);

    public abstract CollationKey getCollationKey(String source);

    public boolean equals(String source, String target);

    public synchronized int getStrength();

    public synchronized void setStrength(int newStrength);

    public synchronized int getDecomposition();

    public synchronized void setDecomposition(int decompositionMode);

    public static synchronized Locale[] getAvailableLocales();

    @Override
    public Object clone();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object that);

    @Override
    public abstract int hashCode();

    protected Collator() {
    }
}
