/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.ref.WeakReference;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import sun.awt.SunHints;

public class RenderingHints implements Map<Object, Object>, Cloneable {

    public abstract static class Key {

        protected Key(int privatekey) {
        }

        public abstract boolean isCompatibleValue(Object val);

        protected final int intKey();

        public final int hashCode();

        public final boolean equals(Object o);
    }

    public static final Key KEY_ANTIALIASING;

    public static final Object VALUE_ANTIALIAS_ON;

    public static final Object VALUE_ANTIALIAS_OFF;

    public static final Object VALUE_ANTIALIAS_DEFAULT;

    public static final Key KEY_RENDERING;

    public static final Object VALUE_RENDER_SPEED;

    public static final Object VALUE_RENDER_QUALITY;

    public static final Object VALUE_RENDER_DEFAULT;

    public static final Key KEY_DITHERING;

    public static final Object VALUE_DITHER_DISABLE;

    public static final Object VALUE_DITHER_ENABLE;

    public static final Object VALUE_DITHER_DEFAULT;

    public static final Key KEY_TEXT_ANTIALIASING;

    public static final Object VALUE_TEXT_ANTIALIAS_ON;

    public static final Object VALUE_TEXT_ANTIALIAS_OFF;

    public static final Object VALUE_TEXT_ANTIALIAS_DEFAULT;

    public static final Object VALUE_TEXT_ANTIALIAS_GASP;

    public static final Object VALUE_TEXT_ANTIALIAS_LCD_HRGB;

    public static final Object VALUE_TEXT_ANTIALIAS_LCD_HBGR;

    public static final Object VALUE_TEXT_ANTIALIAS_LCD_VRGB;

    public static final Object VALUE_TEXT_ANTIALIAS_LCD_VBGR;

    public static final Key KEY_TEXT_LCD_CONTRAST;

    public static final Key KEY_FRACTIONALMETRICS;

    public static final Object VALUE_FRACTIONALMETRICS_OFF;

    public static final Object VALUE_FRACTIONALMETRICS_ON;

    public static final Object VALUE_FRACTIONALMETRICS_DEFAULT;

    public static final Key KEY_INTERPOLATION;

    public static final Object VALUE_INTERPOLATION_NEAREST_NEIGHBOR;

    public static final Object VALUE_INTERPOLATION_BILINEAR;

    public static final Object VALUE_INTERPOLATION_BICUBIC;

    public static final Key KEY_ALPHA_INTERPOLATION;

    public static final Object VALUE_ALPHA_INTERPOLATION_SPEED;

    public static final Object VALUE_ALPHA_INTERPOLATION_QUALITY;

    public static final Object VALUE_ALPHA_INTERPOLATION_DEFAULT;

    public static final Key KEY_COLOR_RENDERING;

    public static final Object VALUE_COLOR_RENDER_SPEED;

    public static final Object VALUE_COLOR_RENDER_QUALITY;

    public static final Object VALUE_COLOR_RENDER_DEFAULT;

    public static final Key KEY_STROKE_CONTROL;

    public static final Object VALUE_STROKE_DEFAULT;

    public static final Object VALUE_STROKE_NORMALIZE;

    public static final Object VALUE_STROKE_PURE;

    public static final Key KEY_RESOLUTION_VARIANT;

    public static final Object VALUE_RESOLUTION_VARIANT_DEFAULT;

    public static final Object VALUE_RESOLUTION_VARIANT_BASE;

    public static final Object VALUE_RESOLUTION_VARIANT_SIZE_FIT;

    public static final Object VALUE_RESOLUTION_VARIANT_DPI_FIT;

    public RenderingHints(Map<Key, ?> init) {
    }

    public RenderingHints(Key key, Object value) {
    }

    public int size();

    public boolean isEmpty();

    public boolean containsKey(Object key);

    public boolean containsValue(Object value);

    public Object get(Object key);

    public Object put(Object key, Object value);

    public void add(RenderingHints hints);

    public void clear();

    public Object remove(Object key);

    public void putAll(Map<?, ?> m);

    public Set<Object> keySet();

    public Collection<Object> values();

    public Set<Map.Entry<Object, Object>> entrySet();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();

    @SuppressWarnings("unchecked")
    public Object clone();

    public String toString();
}
