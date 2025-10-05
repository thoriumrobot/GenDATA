/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2013, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package javax.swing.text;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Toolkit;
    @Positive
import javax.swing.Icon;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class StyleConstants {

    @Positive
    @Interned
    @Positive
    public static final String ComponentElementName;

    @Positive
    @Interned
    @Positive
    public static final String IconElementName;

    @Positive
    public static final Object NameAttribute;

    @Positive
    public static final Object ResolveAttribute;

    @Positive
    public static final Object ModelAttribute;

    @Positive
    public String toString();

    @Positive
    public static final Object BidiLevel;

    @Positive
    public static final Object FontFamily;

    @Positive
    public static final Object Family;

    @Positive
    public static final Object FontSize;

    @Positive
    public static final Object Size;

    @Positive
    public static final Object Bold;

    @Positive
    public static final Object Italic;

    @Positive
    public static final Object Underline;

    @Positive
    public static final Object StrikeThrough;

    @Positive
    public static final Object Superscript;

    @Positive
    public static final Object Subscript;

    @Positive
    public static final Object Foreground;

    @Positive
    public static final Object Background;

    @Positive
    public static final Object ComponentAttribute;

    @Positive
    public static final Object IconAttribute;

    @Positive
    public static final Object ComposedTextAttribute;

    @Positive
    public static final Object FirstLineIndent;

    @Positive
    public static final Object LeftIndent;

    @Positive
    public static final Object RightIndent;

    @Positive
    public static final Object LineSpacing;

    @Positive
    public static final Object SpaceAbove;

    @Positive
    public static final Object SpaceBelow;

    @Positive
    public static final Object Alignment;

    @Positive
    public static final Object TabSet;

    @Positive
    public static final Object Orientation;

    @Positive
    public static final int ALIGN_LEFT;

    @Positive
    public static final int ALIGN_CENTER;

    @Positive
    public static final int ALIGN_RIGHT;

    @Positive
    public static final int ALIGN_JUSTIFIED;

    @Positive
    public static int getBidiLevel(AttributeSet a);

    @Positive
    public static void setBidiLevel(MutableAttributeSet a, int o);

    @Positive
    public static Component getComponent(AttributeSet a);

    @Positive
    public static void setComponent(MutableAttributeSet a, Component c);

    @Positive
    public static Icon getIcon(AttributeSet a);

    @Positive
    public static void setIcon(MutableAttributeSet a, Icon c);

    @Positive
    public static String getFontFamily(AttributeSet a);

    @Positive
    public static void setFontFamily(MutableAttributeSet a, String fam);

    @Positive
    public static int getFontSize(AttributeSet a);

    @Positive
    public static void setFontSize(MutableAttributeSet a, int s);

    @Positive
    public static boolean isBold(AttributeSet a);

    @Positive
    public static void setBold(MutableAttributeSet a, boolean b);

    @Positive
    public static boolean isItalic(AttributeSet a);

    @Positive
    public static void setItalic(MutableAttributeSet a, boolean b);

    @Positive
    public static boolean isUnderline(AttributeSet a);

    @Positive
    public static boolean isStrikeThrough(AttributeSet a);

    @Positive
    public static boolean isSuperscript(AttributeSet a);

    @Positive
    public static boolean isSubscript(AttributeSet a);

    @Positive
    public static void setUnderline(MutableAttributeSet a, boolean b);

    @Positive
    public static void setStrikeThrough(MutableAttributeSet a, boolean b);

    @Positive
    public static void setSuperscript(MutableAttributeSet a, boolean b);

    @Positive
    public static void setSubscript(MutableAttributeSet a, boolean b);

    @Positive
    public static Color getForeground(AttributeSet a);

    @Positive
    public static void setForeground(MutableAttributeSet a, Color fg);

    @Positive
    public static Color getBackground(AttributeSet a);

    @Positive
    public static void setBackground(MutableAttributeSet a, Color fg);

    @Positive
    public static float getFirstLineIndent(AttributeSet a);

    @Positive
    public static void setFirstLineIndent(MutableAttributeSet a, float i);

    @Positive
    public static float getRightIndent(AttributeSet a);

    @Positive
    public static void setRightIndent(MutableAttributeSet a, float i);

    @Positive
    public static float getLeftIndent(AttributeSet a);

    @Positive
    public static void setLeftIndent(MutableAttributeSet a, float i);

    @Positive
    public static float getLineSpacing(AttributeSet a);

    @Positive
    public static void setLineSpacing(MutableAttributeSet a, float i);

    @Positive
    public static float getSpaceAbove(AttributeSet a);

    @Positive
    public static void setSpaceAbove(MutableAttributeSet a, float i);

    @Positive
    public static float getSpaceBelow(AttributeSet a);

    @Positive
    public static void setSpaceBelow(MutableAttributeSet a, float i);

    @Positive
    public static int getAlignment(AttributeSet a);

    @Positive
    public static void setAlignment(MutableAttributeSet a, int align);

    @Positive
    public static TabSet getTabSet(AttributeSet a);

    @Positive
    public static void setTabSet(MutableAttributeSet a, TabSet tabs);

    @Positive
    public static class ParagraphConstants extends StyleConstants implements AttributeSet.ParagraphAttribute {
    @Positive
    }

    @Positive
    public static class CharacterConstants extends StyleConstants implements AttributeSet.CharacterAttribute {
    @Positive
    }

    @Positive
    public static class ColorConstants extends StyleConstants implements AttributeSet.ColorAttribute, AttributeSet.CharacterAttribute {
    @Positive
    }

    @Positive
    public static class FontConstants extends StyleConstants implements AttributeSet.FontAttribute, AttributeSet.CharacterAttribute {
    @Positive
    }
    @Positive
}
