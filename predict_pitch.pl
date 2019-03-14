# predict the unvoiced section of speech
# Victor Apr. 23, 2007

$fvfile = shift @ARGV; #input pitch file, the 1st column is F0
$fufile = shift @ARGV; #ouput pitch file, predicted

# read F0 file with un-voiced
open(FV, "<$fvfile");
$i = 0;
while(<FV>)
{
    chomp;
    @line = split(/\s+/, $_);
    $raw_f0[$i++] = $line[0];
}close(FV);

$vo_id[0] = 0;
$i = 1; $j = 0;
foreach $f (@raw_f0)
{
    if($f > 20)
    {
        $vo_id[$i++] = $j;
    }
    $j++;
}
$sen_len = @raw_f0;
$vo_id[$i] = @raw_f0;
$vo_len = @vo_id;

if ($vo_len > 0)
{
    $f0_m = 0;
    foreach $elem (@raw_f0)
    {
        $f0_m = $f0_m + $elem;
    }
    $f0_m = $f0_m/(@vo_id - 2); # origin the number is @vo_id
    @inf0 = @raw_f0; # Inter_F0
    $inf0[0] = $f0_m; #? why
    $inf0[$sen_len] = $f0_m;
    
    $i = 0;
    for ($j = 0; $j < ($vo_len - 1); $j++)
    {
        $ps = $vo_id[$j+1] - $vo_id[$j] + 1;
        if ($ps < 3)
        {
            next;
        }
        $diff = $inf0[$vo_id[$j]] - $inf0[$vo_id[$j+1]];
        
        undef(@x);
        undef(@y);
        @x = (1 .. $ps);
        if ($diff > 1e-3)
        {
            #Scale_Factor
            $sf = log($diff)/$ps;
            @x = reverse(@x);
            for($k = 0; $k < $ps; $k++)
            {
                $y[$k] = exp($sf * $x[$k]) + $inf0[$vo_id[$j+1]];
            }
        }
        elsif ($diff < -1e-3)
        {
            $sf = log(-$diff)/$ps;
            for($k = 0; $k < $ps; $k++)
            {
                $y[$k] = exp($sf * $x[$k]) + $inf0[$vo_id[$j]];
            }
        }
        else
        {
            for ($k = 0; $k < $ps; $k++)
            {
                $y[$k] = $inf0[$vo_id[$j]];
            }
        }
        
        $h = 0;
        for ($m = $vo_id[$j] + 1; $m < $vo_id[$j+1]; $m++)
        {
            $inf0[$m] = $y[$h];
            $h++;
        }
    }
}

# smoothing using FIR filter
$no = 12; #order = 11
@b = (0.003261, 0.0076237, -0.022349, -0.054296, 0.12573, 0.44003, 0.44003, 0.12573, -0.054296, -0.022349, 0.0076237, 0.003261);
undef(@z);
# 0..0 xxx 0..0
for ($j = 0; $j < $no; $j++)
{
    $z[$j] = 0;
}
for ($j = $no; $j < @inf0 + $no - 1; $j++) # discard the last value of @inf0
{
    $z[$j] = $inf0[$j - $no];
}
for ($j = @inf0 + $no - 1; $j < @inf0 + 2*$no - 1; $j++)
{
    $z[$j] = 0;
}

# y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb) - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
for ($i = 0; $i < @inf0 + $no - 1; $i++)
{
    $inf0s[$i] = 0;
    for ($j = 0; $j < $no; $j++)
    {
        $inf0s[$i] = $inf0s[$i] + $z[$i + $no - $j] * $b[$j];
    }
}

open(FU, ">$fufile");

# discard 0 ~ no/2 and last-no/2 ~ last, remain number: @inf0 - 1
for ($j = $no/2; $j < @inf0 + $no/2 - 1; $j++)
{
    
    printf FU ("%.2f\n", $inf0s[$j]);    
}
close(FU);

