/*
    @Positive
 * Copyright (c) 2002-2019, the original author or authors.
    @Positive
 *
    @Positive
 * This software is distributable under the BSD license. See the terms of the
    @Positive
 * BSD license in the documentation provided with this software.
    @Positive
 *
    @Positive
 * https://opensource.org/licenses/BSD-3-Clause
    @Positive
 */
    @Positive
package jdk.internal.org.jline.utils;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import jdk.internal.org.jline.terminal.Terminal;
    @Positive
import jdk.internal.org.jline.terminal.impl.AbstractWindowsTerminal;
    @Positive
import jdk.internal.org.jline.utils.InfoCmp.Capability;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.BG_COLOR;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.BG_COLOR_EXP;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.FG_COLOR;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.FG_COLOR_EXP;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_BACKGROUND;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_BLINK;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_BOLD;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_CONCEAL;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_CROSSED_OUT;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_FAINT;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_FOREGROUND;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_INVERSE;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_ITALIC;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_UNDERLINE;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.F_HIDDEN;
    @Positive
import static jdk.internal.org.jline.utils.AttributedStyle.MASK;
    @Positive
import static jdk.internal.org.jline.terminal.TerminalBuilder.PROP_DISABLE_ALTERNATE_CHARSET;

    @Positive
public abstract class AttributedCharSequence implements CharSequence {

    @Positive
    public void print(Terminal terminal);

    @Positive
    public void println(Terminal terminal);

    @Positive
    public String toAnsi();

    @Positive
    public String toAnsi(Terminal terminal);

    @Positive
    public String toAnsi(int colors, boolean force256colors);

    @Positive
    public String toAnsi(int colors, boolean force256colors, String altIn, String altOut);

    @Positive
    @Deprecated
    @Positive
    public static int rgbColor(int col);

    @Positive
    @Deprecated
    @Positive
    public static int roundColor(int col, int max);

    @Positive
    @Deprecated
    @Positive
    public static int roundRgbColor(int r, int g, int b, int max);

    @Positive
    public abstract AttributedStyle styleAt(int index);

    @Positive
    int styleCodeAt(int index);

    @Positive
    public boolean isHidden(int index);

    @Positive
    public int runStart(int index);

    @Positive
    public int runLimit(int index);

    @Positive
    @Override
    @Positive
    public abstract AttributedString subSequence(int start, int end);

    @Positive
    public AttributedString substring(int start, int end);

    @Positive
    protected abstract char[] buffer();

    @Positive
    protected abstract int offset();

    @Positive
    @Override
    @Positive
    public char charAt(int index);

    @Positive
    public int codePointAt(int index);

    @Positive
    @Pure
    @Positive
    public boolean contains(char c);

    @Positive
    public int codePointBefore(int index);

    @Positive
    public int codePointCount(int index, int length);

    @Positive
    public int columnLength();

    @Positive
    public AttributedString columnSubSequence(int start, int stop);

    @Positive
    public List<AttributedString> columnSplitLength(int columns);

    @Positive
    public List<AttributedString> columnSplitLength(int columns, boolean includeNewlines, boolean delayLineWrap);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public AttributedString toAttributedString();
    @Positive
}

// CFWR semantic augmentation - variant 1
