/*
 * Copyright (c) 2005, 2018, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.nimbus;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import javax.swing.Painter;
import javax.swing.JComponent;
import javax.swing.UIDefaults;
import javax.swing.UIManager;
import javax.swing.plaf.ColorUIResource;
import javax.swing.plaf.synth.ColorType;
import static javax.swing.plaf.synth.SynthConstants.*;
import javax.swing.plaf.synth.SynthContext;
import javax.swing.plaf.synth.SynthPainter;
import javax.swing.plaf.synth.SynthStyle;
import java.awt.Color;
import java.awt.Font;
import java.awt.Insets;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

@AnnotatedFor({ "interning" })
public final class NimbusStyle extends SynthStyle {

    @Interned
    public static final String LARGE_KEY;

    @Interned
    public static final String SMALL_KEY;

    @Interned
    public static final String MINI_KEY;

    public static final double LARGE_SCALE;

    public static final double SMALL_SCALE;

    public static final double MINI_SCALE;

    @Override
    public void installDefaults(SynthContext ctx);

    @Override
    public Insets getInsets(SynthContext ctx, Insets in);

    @Override
    protected Color getColorForState(SynthContext ctx, ColorType type);

    @Override
    protected Font getFontForState(SynthContext ctx);

    @Override
    public SynthPainter getPainter(SynthContext ctx);

    @Override
    public boolean isOpaque(SynthContext ctx);

    @Override
    public Object get(SynthContext ctx, Object key);

    public Painter<Object> getBackgroundPainter(SynthContext ctx);

    public Painter<Object> getForegroundPainter(SynthContext ctx);

    public Painter<Object> getBorderPainter(SynthContext ctx);

    private final class RuntimeState implements Cloneable {

        @Override
        public String toString();

        @Override
        public RuntimeState clone();
    }

    private static final class Values {
    }

    private static final class CacheKey {

        void init(Object key, int xstate);

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();
    }
}
