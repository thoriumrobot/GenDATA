/*
 * Copyright (c) 2004, 2014, Oracle and/or its affiliates. All rights reserved.
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
package sun.font;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static sun.font.EAttribute.*;
import static java.lang.Math.*;
import java.awt.Font;
import java.awt.Paint;
import java.awt.Toolkit;
import java.awt.font.GraphicAttribute;
import java.awt.font.NumericShaper;
import java.awt.font.TextAttribute;
import java.awt.font.TransformAttribute;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.im.InputMethodHighlight;
import java.io.Serializable;
import java.text.Annotation;
import java.text.AttributedCharacterIterator.Attribute;
import java.util.Map;
import java.util.HashMap;
import java.util.Hashtable;

public final class AttributeValues implements Cloneable {

    public String getFamily();

    public void setFamily(String f);

    public float getWeight();

    public void setWeight(float f);

    public float getWidth();

    public void setWidth(float f);

    public float getPosture();

    public void setPosture(float f);

    public float getSize();

    public void setSize(float f);

    public AffineTransform getTransform();

    public void setTransform(AffineTransform f);

    public void setTransform(TransformAttribute f);

    public int getSuperscript();

    public void setSuperscript(int f);

    public Font getFont();

    public void setFont(Font f);

    public GraphicAttribute getCharReplacement();

    public void setCharReplacement(GraphicAttribute f);

    public Paint getForeground();

    public void setForeground(Paint f);

    public Paint getBackground();

    public void setBackground(Paint f);

    public int getUnderline();

    public void setUnderline(int f);

    public boolean getStrikethrough();

    public void setStrikethrough(boolean f);

    public int getRunDirection();

    public void setRunDirection(int f);

    public int getBidiEmbedding();

    public void setBidiEmbedding(int f);

    public float getJustification();

    public void setJustification(float f);

    public Object getInputMethodHighlight();

    public void setInputMethodHighlight(Annotation f);

    public void setInputMethodHighlight(InputMethodHighlight f);

    public int getInputMethodUnderline();

    public void setInputMethodUnderline(int f);

    public boolean getSwapColors();

    public void setSwapColors(boolean f);

    public NumericShaper getNumericShaping();

    public void setNumericShaping(NumericShaper f);

    public int getKerning();

    public void setKerning(int f);

    public float getTracking();

    public void setTracking(float f);

    public int getLigatures();

    public void setLigatures(int f);

    public AffineTransform getBaselineTransform();

    public AffineTransform getCharTransform();

    public static int getMask(EAttribute att);

    public static int getMask(EAttribute... atts);

    public static final int MASK_ALL;

    public void unsetDefault();

    public void defineAll(int mask);

    public boolean allDefined(int mask);

    public boolean anyDefined(int mask);

    public boolean anyNonDefault(int mask);

    public boolean isDefined(EAttribute a);

    public boolean isNonDefault(EAttribute a);

    public void setDefault(EAttribute a);

    public void unset(EAttribute a);

    public void set(EAttribute a, AttributeValues src);

    public void set(EAttribute a, Object o);

    public Object get(EAttribute a);

    public AttributeValues merge(Map<? extends Attribute, ?> map);

    public AttributeValues merge(Map<? extends Attribute, ?> map, int mask);

    public AttributeValues merge(AttributeValues src);

    public AttributeValues merge(AttributeValues src, int mask);

    public static AttributeValues fromMap(Map<? extends Attribute, ?> map);

    public static AttributeValues fromMap(Map<? extends Attribute, ?> map, int mask);

    public Map<TextAttribute, Object> toMap(Map<TextAttribute, Object> fill);

    public static boolean is16Hashtable(Hashtable<Object, Object> ht);

    public static AttributeValues fromSerializableHashtable(Hashtable<Object, Object> ht);

    public Hashtable<Object, Object> toSerializableHashtable();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object rhs);

    public boolean equals(AttributeValues rhs);

    public AttributeValues clone();

    public String toString();

    public static float getJustification(Map<?, ?> map);

    public static NumericShaper getNumericShaping(Map<?, ?> map);

    public AttributeValues applyIMHighlight();

    @SuppressWarnings("unchecked")
    public static AffineTransform getBaselineTransform(Map<?, ?> map);

    @SuppressWarnings("unchecked")
    public static AffineTransform getCharTransform(Map<?, ?> map);

    public void updateDerivedTransforms();

    public static AffineTransform extractXRotation(AffineTransform tx, boolean andTranslation);

    public static AffineTransform extractYRotation(AffineTransform tx, boolean andTranslation);
}
