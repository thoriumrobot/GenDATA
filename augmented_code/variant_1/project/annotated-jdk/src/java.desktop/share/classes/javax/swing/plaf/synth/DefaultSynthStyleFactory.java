/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class DefaultSynthStyleFactory {
/*
    @Copyright * Positive (c) 2003, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.synth;

    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.plaf.FontUIResource;
    @Positive
import java.awt.Font;
    @Positive
import java.util.*;
    @Positive
import java.util.regex.*;
    @Positive
import sun.swing.plaf.synth.*;
    @Positive
import sun.swing.BakedArrayList;

    @Positive
@AnnotatedFor({ "regex" })
    @Positive
class DefaultSynthStyleFactory extends SynthStyleFactory {

    @Positive
    public static final int NAME;

    @Positive
    public static final int REGION;

    @Positive
    public synchronized void addStyle(DefaultSynthStyle style, @Regex String path, int type) throws PatternSyntaxException;

    @Positive
    public synchronized SynthStyle getStyle(JComponent c, Region id);
    @Positive
}

}