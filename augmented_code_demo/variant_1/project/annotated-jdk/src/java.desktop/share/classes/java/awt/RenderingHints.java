/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

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
import java.lang.ref.WeakReference;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import sun.awt.SunHints;

    @Positive
public class RenderingHints implements Map<Object, Object>, Cloneable {

    @Positive
    public abstract static class Key {

    @Positive
        protected Key(int privatekey) {
    @Positive
        }

    @Positive
        public abstract boolean isCompatibleValue(Object val);

    @Positive
        protected final int intKey();

    @Positive
        public final int hashCode();

    @Positive
        public final boolean equals(Object o);
    @Positive
    }

    @Positive
    public static final Key KEY_ANTIALIASING;

    @Positive
    public static final Object VALUE_ANTIALIAS_ON;

    @Positive
    public static final Object VALUE_ANTIALIAS_OFF;

    @Positive
    public static final Object VALUE_ANTIALIAS_DEFAULT;

    @Positive
    public static final Key KEY_RENDERING;

    @Positive
    public static final Object VALUE_RENDER_SPEED;

    @Positive
    public static final Object VALUE_RENDER_QUALITY;

    @Positive
    public static final Object VALUE_RENDER_DEFAULT;

    @Positive
    public static final Key KEY_DITHERING;

    @Positive
    public static final Object VALUE_DITHER_DISABLE;

    @Positive
    public static final Object VALUE_DITHER_ENABLE;

    @Positive
    public static final Object VALUE_DITHER_DEFAULT;

    @Positive
    public static final Key KEY_TEXT_ANTIALIASING;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_ON;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_OFF;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_DEFAULT;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_GASP;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_LCD_HRGB;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_LCD_HBGR;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_LCD_VRGB;

    @Positive
    public static final Object VALUE_TEXT_ANTIALIAS_LCD_VBGR;

    @Positive
    public static final Key KEY_TEXT_LCD_CONTRAST;

    @Positive
    public static final Key KEY_FRACTIONALMETRICS;

    @Positive
    public static final Object VALUE_FRACTIONALMETRICS_OFF;

    @Positive
    public static final Object VALUE_FRACTIONALMETRICS_ON;

    @Positive
    public static final Object VALUE_FRACTIONALMETRICS_DEFAULT;

    @Positive
    public static final Key KEY_INTERPOLATION;

    @Positive
    public static final Object VALUE_INTERPOLATION_NEAREST_NEIGHBOR;

    @Positive
    public static final Object VALUE_INTERPOLATION_BILINEAR;

    @Positive
    public static final Object VALUE_INTERPOLATION_BICUBIC;

    @Positive
    public static final Key KEY_ALPHA_INTERPOLATION;

    @Positive
    public static final Object VALUE_ALPHA_INTERPOLATION_SPEED;

    @Positive
    public static final Object VALUE_ALPHA_INTERPOLATION_QUALITY;

    @Positive
    public static final Object VALUE_ALPHA_INTERPOLATION_DEFAULT;

    @Positive
    public static final Key KEY_COLOR_RENDERING;

    @Positive
    public static final Object VALUE_COLOR_RENDER_SPEED;

    @Positive
    public static final Object VALUE_COLOR_RENDER_QUALITY;

    @Positive
    public static final Object VALUE_COLOR_RENDER_DEFAULT;

    @Positive
    public static final Key KEY_STROKE_CONTROL;

    @Positive
    public static final Object VALUE_STROKE_DEFAULT;

    @Positive
    public static final Object VALUE_STROKE_NORMALIZE;

    @Positive
    public static final Object VALUE_STROKE_PURE;

    @Positive
    public static final Key KEY_RESOLUTION_VARIANT;

    @Positive
    public static final Object VALUE_RESOLUTION_VARIANT_DEFAULT;

    @Positive
    public static final Object VALUE_RESOLUTION_VARIANT_BASE;

    @Positive
    public static final Object VALUE_RESOLUTION_VARIANT_SIZE_FIT;

    @Positive
    public static final Object VALUE_RESOLUTION_VARIANT_DPI_FIT;

    @Positive
    public RenderingHints(Map<Key, ?> init) {
    @Positive
    }

    @Positive
    public RenderingHints(Key key, Object value) {
    @Positive
    }

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public boolean containsKey(Object key);

    @Positive
    public boolean containsValue(Object value);

    @Positive
    public Object get(Object key);

    @Positive
    public Object put(Object key, Object value);

    @Positive
    public void add(RenderingHints hints);

    @Positive
    public void clear();

    @Positive
    public Object remove(Object key);

    @Positive
    public void putAll(Map<?, ?> m);

    @Positive
    public Set<Object> keySet();

    @Positive
    public Collection<Object> values();

    @Positive
    public Set<Map.Entry<Object, Object>> entrySet();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public Object clone();

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
