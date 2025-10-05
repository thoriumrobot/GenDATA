/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.font.FontRenderContext;
    @Positive
import java.awt.font.GlyphVector;
    @Positive
import java.awt.font.LineMetrics;
    @Positive
import java.awt.font.TextAttribute;
    @Positive
import java.awt.font.TextLayout;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.peer.FontPeer;
    @Positive
import java.io.File;
    @Positive
import java.io.FileOutputStream;
    @Positive
import java.io.FilePermission;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.nio.file.Files;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.text.AttributedCharacterIterator.Attribute;
    @Positive
import java.text.CharacterIterator;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import sun.awt.ComponentFactory;
    @Positive
import sun.font.AttributeMap;
    @Positive
import sun.font.AttributeValues;
    @Positive
import sun.font.CompositeFont;
    @Positive
import sun.font.CoreMetrics;
    @Positive
import sun.font.CreatedFontTracker;
    @Positive
import sun.font.Font2D;
    @Positive
import sun.font.Font2DHandle;
    @Positive
import sun.font.FontAccess;
    @Positive
import sun.font.FontDesignMetrics;
    @Positive
import sun.font.FontLineMetrics;
    @Positive
import sun.font.FontManager;
    @Positive
import sun.font.FontManagerFactory;
    @Positive
import sun.font.FontUtilities;
    @Positive
import sun.font.GlyphLayout;
    @Positive
import sun.font.StandardGlyphVector;
    @Positive
import static sun.font.EAttribute.EBACKGROUND;
    @Positive
import static sun.font.EAttribute.EBIDI_EMBEDDING;
    @Positive
import static sun.font.EAttribute.ECHAR_REPLACEMENT;
    @Positive
import static sun.font.EAttribute.EFAMILY;
    @Positive
import static sun.font.EAttribute.EFONT;
    @Positive
import static sun.font.EAttribute.EFOREGROUND;
    @Positive
import static sun.font.EAttribute.EINPUT_METHOD_HIGHLIGHT;
    @Positive
import static sun.font.EAttribute.EINPUT_METHOD_UNDERLINE;
    @Positive
import static sun.font.EAttribute.EJUSTIFICATION;
    @Positive
import static sun.font.EAttribute.EKERNING;
    @Positive
import static sun.font.EAttribute.ELIGATURES;
    @Positive
import static sun.font.EAttribute.ENUMERIC_SHAPING;
    @Positive
import static sun.font.EAttribute.EPOSTURE;
    @Positive
import static sun.font.EAttribute.ERUN_DIRECTION;
    @Positive
import static sun.font.EAttribute.ESIZE;
    @Positive
import static sun.font.EAttribute.ESTRIKETHROUGH;
    @Positive
import static sun.font.EAttribute.ESUPERSCRIPT;
    @Positive
import static sun.font.EAttribute.ESWAP_COLORS;
    @Positive
import static sun.font.EAttribute.ETRACKING;
    @Positive
import static sun.font.EAttribute.ETRANSFORM;
    @Positive
import static sun.font.EAttribute.EUNDERLINE;
    @Positive
import static sun.font.EAttribute.EWEIGHT;
    @Positive
import static sun.font.EAttribute.EWIDTH;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class Font implements java.io.Serializable {

    @Positive
    private static class FontAccessImpl extends FontAccess {

    @Positive
        public Font2D getFont2D(Font font);

    @Positive
        public void setFont2D(Font font, Font2DHandle handle);

    @Positive
        public void setCreatedFont(Font font);

    @Positive
        public boolean isCreatedFont(Font font);

    @Positive
        @Override
    @Positive
        public FontPeer getFontPeer(final Font font);
    @Positive
    }

    @Positive
    @Interned
    @Positive
    public static final String DIALOG;

    @Positive
    @Interned
    @Positive
    public static final String DIALOG_INPUT;

    @Positive
    @Interned
    @Positive
    public static final String SANS_SERIF;

    @Positive
    @Interned
    @Positive
    public static final String SERIF;

    @Positive
    @Interned
    @Positive
    public static final String MONOSPACED;

    @Positive
    public static final int PLAIN;

    @Positive
    public static final int BOLD;

    @Positive
    public static final int ITALIC;

    @Positive
    public static final int ROMAN_BASELINE;

    @Positive
    public static final int CENTER_BASELINE;

    @Positive
    public static final int HANGING_BASELINE;

    @Positive
    public static final int TRUETYPE_FONT;

    @Positive
    public static final int TYPE1_FONT;

    @Positive
    protected String name;

    @Positive
    protected int style;

    @Positive
    protected int size;

    @Positive
    protected float pointSize;

    @Positive
    public Font(String name, int style, int size) {
    @Positive
    }

    @Positive
    public Font(Map<? extends Attribute, ?> attributes) {
    @Positive
    }

    @Positive
    protected Font(Font font) {
    @Positive
    }

    @Positive
    public static boolean textRequiresLayout(char[] chars, int start, int end);

    @Positive
    public static Font getFont(Map<? extends Attribute, ?> attributes);

    @Positive
    public static Font[] createFonts(InputStream fontStream) throws FontFormatException, IOException;

    @Positive
    public static Font[] createFonts(File fontFile) throws FontFormatException, IOException;

    @Positive
    public static Font createFont(int fontFormat, InputStream fontStream) throws java.awt.FontFormatException, java.io.IOException;

    @Positive
    public static Font createFont(int fontFormat, File fontFile) throws java.awt.FontFormatException, java.io.IOException;

    @Positive
    public AffineTransform getTransform();

    @Positive
    public String getFamily();

    @Positive
    final String getFamily_NoClientCode();

    @Positive
    public String getFamily(Locale l);

    @Positive
    public String getPSName();

    @Positive
    public String getName();

    @Positive
    public String getFontName();

    @Positive
    public String getFontName(Locale l);

    @Positive
    public int getStyle();

    @Positive
    public int getSize();

    @Positive
    public float getSize2D();

    @Positive
    public boolean isPlain();

    @Positive
    public boolean isBold();

    @Positive
    public boolean isItalic();

    @Positive
    public boolean isTransformed();

    @Positive
    public boolean hasLayoutAttributes();

    @Positive
    public static Font getFont(String nm);

    @Positive
    public static Font decode(String str);

    @Positive
    public static Font getFont(String nm, Font font);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public String toString();

    @Positive
    public int getNumGlyphs();

    @Positive
    public int getMissingGlyphCode();

    @Positive
    public byte getBaselineFor(char c);

    @Positive
    public Map<TextAttribute, ?> getAttributes();

    @Positive
    public Attribute[] getAvailableAttributes();

    @Positive
    public Font deriveFont(int style, float size);

    @Positive
    public Font deriveFont(int style, AffineTransform trans);

    @Positive
    public Font deriveFont(float size);

    @Positive
    public Font deriveFont(AffineTransform trans);

    @Positive
    public Font deriveFont(int style);

    @Positive
    public Font deriveFont(Map<? extends Attribute, ?> attributes);

    @Positive
    public boolean canDisplay(char c);

    @Positive
    public boolean canDisplay(int codePoint);

    @Positive
    public int canDisplayUpTo(String str);

    @Positive
    public int canDisplayUpTo(char[] text, int start, int limit);

    @Positive
    public int canDisplayUpTo(CharacterIterator iter, int start, int limit);

    @Positive
    public float getItalicAngle();

    @Positive
    public boolean hasUniformLineMetrics();

    @Positive
    public LineMetrics getLineMetrics(String str, FontRenderContext frc);

    @Positive
    public LineMetrics getLineMetrics(String str, int beginIndex, int limit, FontRenderContext frc);

    @Positive
    public LineMetrics getLineMetrics(char[] chars, int beginIndex, int limit, FontRenderContext frc);

    @Positive
    public LineMetrics getLineMetrics(CharacterIterator ci, int beginIndex, int limit, FontRenderContext frc);

    @Positive
    public Rectangle2D getStringBounds(String str, FontRenderContext frc);

    @Positive
    public Rectangle2D getStringBounds(String str, int beginIndex, int limit, FontRenderContext frc);

    @Positive
    public Rectangle2D getStringBounds(char[] chars, int beginIndex, int limit, FontRenderContext frc);

    @Positive
    public Rectangle2D getStringBounds(CharacterIterator ci, int beginIndex, int limit, FontRenderContext frc);

    @Positive
    public Rectangle2D getMaxCharBounds(FontRenderContext frc);

    @Positive
    public GlyphVector createGlyphVector(FontRenderContext frc, String str);

    @Positive
    public GlyphVector createGlyphVector(FontRenderContext frc, char[] chars);

    @Positive
    public GlyphVector createGlyphVector(FontRenderContext frc, CharacterIterator ci);

    @Positive
    public GlyphVector createGlyphVector(FontRenderContext frc, int[] glyphCodes);

    @Positive
    public GlyphVector layoutGlyphVector(FontRenderContext frc, char[] text, int start, int limit, int flags);

    @Positive
    public static final int LAYOUT_LEFT_TO_RIGHT;

    @Positive
    public static final int LAYOUT_RIGHT_TO_LEFT;

    @Positive
    public static final int LAYOUT_NO_START_CONTEXT;

    @Positive
    public static final int LAYOUT_NO_LIMIT_CONTEXT;
    @Positive
}

// CFWR semantic augmentation - variant 1
