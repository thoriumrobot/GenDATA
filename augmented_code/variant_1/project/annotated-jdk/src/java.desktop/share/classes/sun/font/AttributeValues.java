/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2004, 2014, Oracle and/or its affiliates. All rights reserved.
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
package sun.font;

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
import static sun.font.EAttribute.*;
    @Positive
import static java.lang.Math.*;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Paint;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.font.GraphicAttribute;
    @Positive
import java.awt.font.NumericShaper;
    @Positive
import java.awt.font.TextAttribute;
    @Positive
import java.awt.font.TransformAttribute;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.NoninvertibleTransformException;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.im.InputMethodHighlight;
    @Positive
import java.io.Serializable;
    @Positive
import java.text.Annotation;
    @Positive
import java.text.AttributedCharacterIterator.Attribute;
    @Positive
import java.util.Map;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Hashtable;

    @Positive
public final class AttributeValues implements Cloneable {

    @Positive
    public String getFamily();

    @Positive
    public void setFamily(String f);

    @Positive
    public float getWeight();

    @Positive
    public void setWeight(float f);

    @Positive
    public float getWidth();

    @Positive
    public void setWidth(float f);

    @Positive
    public float getPosture();

    @Positive
    public void setPosture(float f);

    @Positive
    public float getSize();

    @Positive
    public void setSize(float f);

    @Positive
    public AffineTransform getTransform();

    @Positive
    public void setTransform(AffineTransform f);

    @Positive
    public void setTransform(TransformAttribute f);

    @Positive
    public int getSuperscript();

    @Positive
    public void setSuperscript(int f);

    @Positive
    public Font getFont();

    @Positive
    public void setFont(Font f);

    @Positive
    public GraphicAttribute getCharReplacement();

    @Positive
    public void setCharReplacement(GraphicAttribute f);

    @Positive
    public Paint getForeground();

    @Positive
    public void setForeground(Paint f);

    @Positive
    public Paint getBackground();

    @Positive
    public void setBackground(Paint f);

    @Positive
    public int getUnderline();

    @Positive
    public void setUnderline(int f);

    @Positive
    public boolean getStrikethrough();

    @Positive
    public void setStrikethrough(boolean f);

    @Positive
    public int getRunDirection();

    @Positive
    public void setRunDirection(int f);

    @Positive
    public int getBidiEmbedding();

    @Positive
    public void setBidiEmbedding(int f);

    @Positive
    public float getJustification();

    @Positive
    public void setJustification(float f);

    @Positive
    public Object getInputMethodHighlight();

    @Positive
    public void setInputMethodHighlight(Annotation f);

    @Positive
    public void setInputMethodHighlight(InputMethodHighlight f);

    @Positive
    public int getInputMethodUnderline();

    @Positive
    public void setInputMethodUnderline(int f);

    @Positive
    public boolean getSwapColors();

    @Positive
    public void setSwapColors(boolean f);

    @Positive
    public NumericShaper getNumericShaping();

    @Positive
    public void setNumericShaping(NumericShaper f);

    @Positive
    public int getKerning();

    @Positive
    public void setKerning(int f);

    @Positive
    public float getTracking();

    @Positive
    public void setTracking(float f);

    @Positive
    public int getLigatures();

    @Positive
    public void setLigatures(int f);

    @Positive
    public AffineTransform getBaselineTransform();

    @Positive
    public AffineTransform getCharTransform();

    @Positive
    public static int getMask(EAttribute att);

    @Positive
    public static int getMask(EAttribute... atts);

    @Positive
    public static final int MASK_ALL;

    @Positive
    public void unsetDefault();

    @Positive
    public void defineAll(int mask);

    @Positive
    public boolean allDefined(int mask);

    @Positive
    public boolean anyDefined(int mask);

    @Positive
    public boolean anyNonDefault(int mask);

    @Positive
    public boolean isDefined(EAttribute a);

    @Positive
    public boolean isNonDefault(EAttribute a);

    @Positive
    public void setDefault(EAttribute a);

    @Positive
    public void unset(EAttribute a);

    @Positive
    public void set(EAttribute a, AttributeValues src);

    @Positive
    public void set(EAttribute a, Object o);

    @Positive
    public Object get(EAttribute a);

    @Positive
    public AttributeValues merge(Map<? extends Attribute, ?> map);

    @Positive
    public AttributeValues merge(Map<? extends Attribute, ?> map, int mask);

    @Positive
    public AttributeValues merge(AttributeValues src);

    @Positive
    public AttributeValues merge(AttributeValues src, int mask);

    @Positive
    public static AttributeValues fromMap(Map<? extends Attribute, ?> map);

    @Positive
    public static AttributeValues fromMap(Map<? extends Attribute, ?> map, int mask);

    @Positive
    public Map<TextAttribute, Object> toMap(Map<TextAttribute, Object> fill);

    @Positive
    public static boolean is16Hashtable(Hashtable<Object, Object> ht);

    @Positive
    public static AttributeValues fromSerializableHashtable(Hashtable<Object, Object> ht);

    @Positive
    public Hashtable<Object, Object> toSerializableHashtable();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object rhs);

    @Positive
    public boolean equals(AttributeValues rhs);

    @Positive
    public AttributeValues clone();

    @Positive
    public String toString();

    @Positive
    public static float getJustification(Map<?, ?> map);

    @Positive
    public static NumericShaper getNumericShaping(Map<?, ?> map);

    @Positive
    public AttributeValues applyIMHighlight();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static AffineTransform getBaselineTransform(Map<?, ?> map);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static AffineTransform getCharTransform(Map<?, ?> map);

    @Positive
    public void updateDerivedTransforms();

    @Positive
    public static AffineTransform extractXRotation(AffineTransform tx, boolean andTranslation);

    @Positive
    public static AffineTransform extractYRotation(AffineTransform tx, boolean andTranslation);
    @Positive
}
