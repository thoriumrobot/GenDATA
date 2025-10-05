/*
    @Positive
 * Copyright (c) 2005, 2018, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.nimbus;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.swing.Painter;
    @Positive
import javax.swing.JComponent;
    @Positive
import javax.swing.UIDefaults;
    @Positive
import javax.swing.UIManager;
    @Positive
import javax.swing.plaf.ColorUIResource;
    @Positive
import javax.swing.plaf.synth.ColorType;
    @Positive
import static javax.swing.plaf.synth.SynthConstants.*;
    @Positive
import javax.swing.plaf.synth.SynthContext;
    @Positive
import javax.swing.plaf.synth.SynthPainter;
    @Positive
import javax.swing.plaf.synth.SynthStyle;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Insets;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.TreeMap;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public final class NimbusStyle extends SynthStyle {

    @Positive
    @Interned
    @Positive
    public static final String LARGE_KEY;

    @Positive
    @Interned
    @Positive
    public static final String SMALL_KEY;

    @Positive
    @Interned
    @Positive
    public static final String MINI_KEY;

    @Positive
    public static final double LARGE_SCALE;

    @Positive
    public static final double SMALL_SCALE;

    @Positive
    public static final double MINI_SCALE;

    @Positive
    @Override
    @Positive
    public void installDefaults(SynthContext ctx);

    @Positive
    @Override
    @Positive
    public Insets getInsets(SynthContext ctx, Insets in);

    @Positive
    @Override
    @Positive
    protected Color getColorForState(SynthContext ctx, ColorType type);

    @Positive
    @Override
    @Positive
    protected Font getFontForState(SynthContext ctx);

    @Positive
    @Override
    @Positive
    public SynthPainter getPainter(SynthContext ctx);

    @Positive
    @Override
    @Positive
    public boolean isOpaque(SynthContext ctx);

    @Positive
    @Override
    @Positive
    public Object get(SynthContext ctx, Object key);

    @Positive
    public Painter<Object> getBackgroundPainter(SynthContext ctx);

    @Positive
    public Painter<Object> getForegroundPainter(SynthContext ctx);

    @Positive
    public Painter<Object> getBorderPainter(SynthContext ctx);

    @Positive
    private final class RuntimeState implements Cloneable {

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public RuntimeState clone();
    @Positive
    }

    @Positive
    private static final class Values {
    @Positive
    }

    @Positive
    private static final class CacheKey {

    @Positive
        void init(Object key, int xstate);

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
